import asyncio
import base64
import io
import os
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path

from a2a.server.tasks import TaskUpdater
from a2a.types import (
    FilePart, FileWithBytes, Message, Part, TaskState, TextPart
)
from a2a.utils import get_message_text, new_agent_text_message
from messenger import Messenger
from openai import AsyncOpenAI

client = AsyncOpenAI(
    api_key=os.environ["OPENROUTER_API_KEY"],
    base_url="https://openrouter.ai/api/v1",
)

MODEL = "qwen/qwen3.6-plus:free"


class Agent:
    def __init__(self):
        self.messenger = Messenger()
        self._workdir: Path | None = None
        self._workdir_obj = None  # держим tempfile объект живым
        self._competition_id: str | None = None
        self._instructions: str = ""

    async def run(self, message: Message, updater: TaskUpdater) -> None:
        text = get_message_text(message)

        # Если green просит валидацию — обрабатываем отдельно
        if "validate" in text.lower() and self._workdir:
            await self._handle_validation(message, updater)
            return

        # Иначе — основной флоу: получить датасет и решить задачу
        await self._handle_main_task(message, updater)

    async def _handle_validation(self, message: Message, updater: TaskUpdater) -> None:
        """Валидируем submission.csv который прислал green-агент."""
        await updater.update_status(
            TaskState.working,
            new_agent_text_message("Validating submission..."),
        )

        submission_data = None
        for part in message.parts:
            if isinstance(part.root, FilePart):
                file_data = part.root.file
                if isinstance(file_data, FileWithBytes):
                    submission_data = base64.b64decode(file_data.bytes)
                    break

        if not submission_data:
            await updater.add_artifact(
                parts=[Part(root=TextPart(text="Error: No submission file provided"))],
                name="validation_result",
            )
            return

        # Сохраняем и проверяем базово
        val_path = self._workdir / "validate_input.csv"
        val_path.write_bytes(submission_data)

        try:
            import pandas as pd
            df = pd.read_csv(val_path)
            result = f"Valid submission: {len(df)} rows, columns: {list(df.columns)}"
        except Exception as e:
            result = f"Invalid submission: {e}"

        await updater.add_artifact(
            parts=[Part(root=TextPart(text=result))],
            name="validation_result",
        )

    async def _handle_main_task(self, message: Message, updater: TaskUpdater) -> None:
        await updater.update_status(
            TaskState.working,
            new_agent_text_message("Received task, extracting data..."),
        )

        instructions = ""
        tar_bytes = None

        for part in message.parts:
            if isinstance(part.root, TextPart):
                instructions += part.root.text + "\n"
            elif isinstance(part.root, FilePart):
                file_data = part.root.file
                if isinstance(file_data, FileWithBytes):
                    tar_bytes = base64.b64decode(file_data.bytes)

        if not tar_bytes:
            await updater.update_status(
                TaskState.failed,
                new_agent_text_message("No dataset tar.gz found in message"),
            )
            return

        # Создаём постоянную tmpdir (живёт пока живёт агент)
        self._workdir_obj = tempfile.TemporaryDirectory()
        self._workdir = Path(self._workdir_obj.name)
        self._instructions = instructions

        # Распаковываем
        try:
            with tarfile.open(fileobj=io.BytesIO(tar_bytes), mode="r:gz") as tar:
                tar.extractall(self._workdir)
        except Exception as e:
            await updater.update_status(
                TaskState.failed,
                new_agent_text_message(f"Failed to extract tar: {e}"),
            )
            return

        data_dir = self._workdir / "home" / "data"
        if not data_dir.exists():
            data_dir = self._workdir

        files_list = []
        for f in sorted(data_dir.rglob("*"))[:30]:
            if f.is_file():
                files_list.append(str(f.relative_to(self._workdir)))
        files_info = "\n".join(files_list)

        await updater.update_status(
            TaskState.working,
            new_agent_text_message(f"Files extracted:\n{files_info}\n\nGenerating solution..."),
        )

        # Читаем превью данных
        sample_info = ""
        for f in data_dir.rglob("sample_submission*"):
            try:
                sample_info = f"\nSample submission:\n{f.read_text()[:500]}"
                break
            except Exception:
                pass

        train_info = ""
        for f in data_dir.rglob("train.csv"):
            try:
                train_info = f"\nTrain preview:\n{f.read_text()[:1000]}"
                break
            except Exception:
                pass

        # Генерируем код
        code = await self._generate_code(files_info, sample_info, train_info)

        # Запускаем
        submission_bytes = await self._run_with_retry(code, updater)
        if submission_bytes is None:
            return

        await updater.update_status(
            TaskState.working,
            new_agent_text_message("Solution done! Submitting..."),
        )

        await updater.add_artifact(
            parts=[Part(root=FilePart(
                file=FileWithBytes(
                    bytes=base64.b64encode(submission_bytes).decode("ascii"),
                    name="submission.csv",
                    mime_type="text/csv",
                )
            ))],
            name="submission",
        )

    async def _generate_code(self, files_info: str, sample_info: str, train_info: str) -> str:
        prompt = f"""You are an expert ML engineer solving a Kaggle competition.

COMPETITION INSTRUCTIONS:
{self._instructions}

FILES (relative to /workdir):
{files_info}
{train_info}
{sample_info}

Write a complete Python script that:
1. Loads data from /workdir/home/data/
2. Trains a simple scikit-learn model (RandomForest or LogisticRegression)
3. Makes predictions on test data
4. Saves to /workdir/submission.csv in the exact format of sample_submission.csv

Rules:
- Use only: pandas, numpy, scikit-learn
- Handle missing values with fillna or SimpleImputer
- Save submission.csv to exactly /workdir/submission.csv
- Output ONLY raw Python code, no markdown, no explanation
"""
        response = await client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=3000,
        )
        code = response.choices[0].message.content.strip()
        for marker in ["```python", "```"]:
            code = code.replace(marker, "")
        return code.strip()

    async def _run_code(self, code: str) -> subprocess.CompletedProcess:
        script_path = self._workdir / "solution.py"
        script_path.write_text(code)

        await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: subprocess.run(
                [sys.executable, "-m", "pip", "install", "-q",
                 "scikit-learn", "pandas", "numpy"],
                capture_output=True,
            ),
        )

        return await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: subprocess.run(
                [sys.executable, str(script_path)],
                capture_output=True,
                text=True,
                timeout=300,
                env={**os.environ, "WORKDIR": str(self._workdir)},
            ),
        )

    async def _run_with_retry(self, code: str, updater: TaskUpdater) -> bytes | None:
        result = await self._run_code(code)
        submission_path = self._workdir / "submission.csv"

        if result.returncode == 0 and submission_path.exists():
            return submission_path.read_bytes()

        # Пробуем починить
        await updater.update_status(
            TaskState.working,
            new_agent_text_message("First attempt failed, fixing..."),
        )

        fix_prompt = f"""This ML script failed:

ERROR:
{result.stderr[-2000:]}

STDOUT:
{result.stdout[-500:]}

ORIGINAL CODE:
{code}

Fix it. Data is in /workdir/home/data/, save to /workdir/submission.csv.
Output ONLY raw Python code, no markdown.
"""
        fix_response = await client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": fix_prompt}],
            max_tokens=3000,
        )
        fixed_code = fix_response.choices[0].message.content.strip()
        for marker in ["```python", "```"]:
            fixed_code = fixed_code.replace(marker, "")
        fixed_code = fixed_code.strip()

        result2 = await self._run_code(fixed_code)

        if result2.returncode == 0 and submission_path.exists():
            return submission_path.read_bytes()

        await updater.update_status(
            TaskState.failed,
            new_agent_text_message(f"Failed after retry.\nError: {result2.stderr[-1000:]}"),
        )
        return None