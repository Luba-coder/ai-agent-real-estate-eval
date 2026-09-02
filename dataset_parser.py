# dataset_parser.py
import pandas as pd
from typing import Dict, List, Optional
from pathlib import Path


class DatasetParser:
    """Простой парсер датасета для оценки"""

    def __init__(self):
        self.required_columns = ['question']  # Только question обязательно
        self.optional_columns = ['expected_response', 'expected_tools']

    def load_dataset(self, file_path: str) -> Optional[pd.DataFrame]:
        """Загрузить датасет из Excel или CSV файла"""
        try:
            # Определяем расширение файла
            file_extension = Path(file_path).suffix.lower()

            # Загружаем файл
            if file_extension in ['.xlsx', '.xls']:
                df = pd.read_excel(file_path)
            elif file_extension == '.csv':
                df = pd.read_csv(file_path)
            else:
                print(f"Неподдерживаемый формат файла: {file_extension}")
                return None

            # Проверяем наличие обязательных колонок
            missing_columns = [
                col for col in self.required_columns if col not in df.columns]
            if missing_columns:
                print(f"Отсутствуют обязательные колонки: {missing_columns}")
                print(f"Найденные колонки: {list(df.columns)}")
                return None

            # Удаляем строки с пустыми значениями в обязательных колонках
            df_clean = df.dropna(subset=self.required_columns)

            print(f"Загружен датасет с {len(df_clean)} вопросами")

            # Показываем информацию о найденных опциональных колонках
            found_optional = [
                col for col in self.optional_columns if col in df_clean.columns]
            if found_optional:
                print(f"Найдены опциональные колонки: {found_optional}")

            return df_clean

        except Exception as e:
            print(f"Ошибка при загрузке файла: {e}")
            return None

    def _parse_tools_field(self, tools_value) -> List[str]:
        """
        Парсит поле с тулами в список

        Поддерживаемые форматы:
        - "tool1, tool2, tool3" (comma-separated)
        - "tool1; tool2; tool3" (semicolon-separated)
        - ["tool1", "tool2", "tool3"] (уже список)
        - NaN / None -> пустой список
        """
        # Если пустое значение
        if pd.isna(tools_value) or tools_value is None or tools_value == "":
            return []

        # Если уже список
        if isinstance(tools_value, list):
            return [str(tool).strip() for tool in tools_value if str(tool).strip()]

        # Если строка
        tools_str = str(tools_value).strip()
        if not tools_str:
            return []

        # Пробуем разделить по запятой или точке с запятой
        if ',' in tools_str:
            tools = [t.strip() for t in tools_str.split(',')]
        elif ';' in tools_str:
            tools = [t.strip() for t in tools_str.split(';')]
        else:
            # Один тул
            tools = [tools_str]

        # Убираем пустые строки
        return [t for t in tools if t]

    def get_questions(self, df: pd.DataFrame) -> List[str]:
        """Получить список всех вопросов"""
        return df['question'].tolist()

    def get_expected_responses(self, df: pd.DataFrame) -> List[str]:
        """Получить список ожидаемых ответов, заменяя nan на пустую строку"""
        if 'expected_response' not in df.columns:
            return [""] * len(df)
        return df['expected_response'].fillna("").astype(str).tolist()

    def get_expected_tools(self, df: pd.DataFrame) -> List[List[str]]:
        """
        Получить список ожидаемых тулов для каждого вопроса

        Returns:
            Список списков тулов (например, [["tool1", "tool2"], ["tool3"], []])
        """
        if 'expected_tools' not in df.columns:
            return [[] for _ in range(len(df))]

        return [self._parse_tools_field(tools) for tools in df['expected_tools']]

    def get_question_response_pairs(self, df: pd.DataFrame) -> List[Dict[str, any]]:
        """
        Получить список пар вопрос-ответ с ожидаемыми тулами

        Returns:
            Список словарей с ключами 'question', 'expected_response', 'expected_tools'
        """
        pairs = []
        expected_tools_list = self.get_expected_tools(df)

        for i, row in df.iterrows():
            expected_response = row.get('expected_response', '')
            if pd.isna(expected_response) or str(expected_response).strip().lower() in ['nan', 'none', '']:
                expected_response = None
            else:
                expected_response = str(expected_response).strip()
            pair = {
                'question': row['question'],
                'expected_response': expected_response,
                'expected_tools': expected_tools_list[i] if i < len(expected_tools_list) else []
            }
            pairs.append(pair)

        return pairs

    def validate_dataset(self, df: pd.DataFrame) -> Dict[str, any]:
        """
        Валидация датасета

        Returns:
            Словарь с информацией о датасете
        """
        info = {
            'total_rows': len(df),
            'valid_pairs': 0,
            'empty_questions': 0,
            'empty_responses': 0,
            'empty_tools': 0,
            'avg_question_length': 0,
            'avg_response_length': 0,
            'avg_tools_count': 0,
            'unique_tools': set(),
            'has_expected_response_column': 'expected_response' in df.columns,
            'has_expected_tools_column': 'expected_tools' in df.columns
        }

        valid_pairs = 0
        question_lengths = []
        response_lengths = []
        tools_counts = []

        expected_tools_list = self.get_expected_tools(df)

        for i, row in df.iterrows():
            question = str(row['question']).strip()

            if not question or question == 'nan':
                info['empty_questions'] += 1
                continue

            # Проверяем expected_response если колонка есть
            if 'expected_response' in df.columns:
                response = str(row['expected_response']).strip()
                if not response or response == 'nan':
                    info['empty_responses'] += 1
                else:
                    response_lengths.append(len(response))

            # Проверяем expected_tools
            tools = expected_tools_list[i] if i < len(
                expected_tools_list) else []
            if not tools:
                info['empty_tools'] += 1
            else:
                tools_counts.append(len(tools))
                info['unique_tools'].update(tools)

            valid_pairs += 1
            question_lengths.append(len(question))

        info['valid_pairs'] = valid_pairs
        if question_lengths:
            info['avg_question_length'] = sum(
                question_lengths) / len(question_lengths)
        if response_lengths:
            info['avg_response_length'] = sum(
                response_lengths) / len(response_lengths)
        if tools_counts:
            info['avg_tools_count'] = sum(tools_counts) / len(tools_counts)

        return info

    def preview_dataset(self, df: pd.DataFrame, n: int = 3):
        """Показать первые n примеров из датасета"""
        print("\n=== Превью датасета ===")
        expected_tools_list = self.get_expected_tools(df)

        for i, row in df.head(n).iterrows():
            print(f"\nПример {i+1}:")
            print(f"Вопрос: {row['question']}")

            # Ожидаемый ответ
            if 'expected_response' in df.columns:
                expected_response = row.get('expected_response')
                if expected_response is not None and not pd.isna(expected_response):
                    response_preview = str(expected_response)[:100]
                    if len(str(expected_response)) > 100:
                        response_preview += "..."
                    print(f"Ожидаемый ответ: {response_preview}")
                else:
                    print("Ожидаемый ответ: Не задан")

            # Ожидаемые тулы
            if 'expected_tools' in df.columns:
                tools = expected_tools_list[i] if i < len(
                    expected_tools_list) else []
                if tools:
                    print(f"Ожидаемые тулы: {', '.join(tools)}")
                else:
                    print("Ожидаемые тулы: Не заданы")

        print("=" * 50)
