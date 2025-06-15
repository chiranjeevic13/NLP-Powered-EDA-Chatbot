import pandas as pd
import spacy
from typing import Union, Dict, Any, List, Tuple
import re
from fuzzywuzzy import process
import numpy as np

class UltraAdvancedMathModule:
    def __init__(self, df: pd.DataFrame):
        self.df = df
        try:
            self.nlp = spacy.load('en_core_web_sm')
        except OSError:
            from spacy.cli import download
            download('en_core_web_sm')
            self.nlp = spacy.load('en_core_web_sm')
        self._preprocess_columns()

    def _preprocess_columns(self):
        self.column_metadata = {}
        for col in self.df.columns:
            self.column_metadata[col] = {
                'dtype': str(self.df[col].dtype),
                'min': self.df[col].min() if pd.api.types.is_numeric_dtype(self.df[col]) else None,
                'max': self.df[col].max() if pd.api.types.is_numeric_dtype(self.df[col]) else None,
                'unique_count': self.df[col].nunique(),
                'semantic_variations': self._generate_semantic_variations(col)
            }

    def _generate_semantic_variations(self, column_name: str) -> List[str]:
        semantic_map = {
            'salary': ['income', 'earnings', 'pay', 'compensation'],
            'age': ['years', 'year old', 'age group'],
            'experience': ['tenure', 'work years', 'professional experience'],
            'score': ['rating', 'performance', 'evaluation']
        }
        variations = [column_name.lower()]
        for key, synonyms in semantic_map.items():
            if key in column_name.lower():
                variations.extend(synonyms)
        return list(set(variations))

    def advanced_column_matching(self, query: str) -> List[str]:
        doc = self.nlp(query.lower())
        matched_columns = []
        for col, metadata in self.column_metadata.items():
            for variation in metadata['semantic_variations']:
                if variation in query.lower():
                    matched_columns.append(col)
        if not matched_columns:
            column_names = list(self.column_metadata.keys())
            fuzzy_matches = process.extract(query, column_names, limit=3)
            matched_columns = [match[0] for match in fuzzy_matches if match[1] > 70]
        if not matched_columns:
            matched_columns = self._nlp_semantic_matching(doc)
        return list(set(matched_columns))

    def _nlp_semantic_matching(self, doc) -> List[str]:
        semantic_matches = []
        key_tokens = [token.lemma_ for token in doc if token.pos_ in ['NOUN', 'PROPN'] and len(token.lemma_) > 2]
        for col, metadata in self.column_metadata.items():
            for token in key_tokens:
                if (token in col.lower() or any(token in var for var in metadata['semantic_variations'])):
                    semantic_matches.append(col)
        return semantic_matches

    def execute_math_query(self, query: str) -> Union[float, pd.DataFrame, str, List[Any]]:
        try:
            query_log = {'original_query': query, 'timestamp': pd.Timestamp.now()}
            normalized_query = self._normalize_query(query)
            columns = self.advanced_column_matching(normalized_query)
            if not columns:
                return self._generate_error_response("No matching columns found", query_log, suggestion="Try using more specific column names or variations")
            operations = self._advanced_operation_detection(normalized_query)
            results = self._process_math_operations(columns, operations, query_log)
            return results
        except Exception as e:
            return self._generate_error_response(f"Unexpected error: {str(e)}", query_log)

    def _normalize_query(self, query: str) -> str:
        normalized = query.lower().strip()
        normalized = re.sub(r'\s+', ' ', normalized)
        replacements = {'calculate': '', 'find': '', 'show': '', 'get': ''}
        for old, new in replacements.items():
            normalized = normalized.replace(old, new)
        return normalized.strip()

    def _advanced_operation_detection(self, query: str) -> List[str]:
        operation_map = {
            'sum': ['total', 'sum', 'add', 'aggregate'],
            'mean': ['average', 'mean', 'avg', 'central'],
            'median': ['median', 'midpoint', 'middle'],
            'std': ['standard deviation', 'spread', 'variance'],
            'min': ['minimum', 'lowest', 'smallest'],
            'max': ['maximum', 'highest', 'largest'],
            'range': ['range', 'difference', 'gap'],
            'percentile': ['percentile', 'quantile'],
            'count': ['count', 'number', 'quantity'],
            'nth_max': ['nth maximum', 'nth highest'],
            'nth_min': ['nth minimum', 'nth lowest']
        }
        detected_ops = []
        for op, keywords in operation_map.items():
            if any(keyword in query for keyword in keywords):
                detected_ops.append(op)
        return detected_ops if detected_ops else ['mean', 'std', 'min', 'max', 'count']

    def _process_math_operations(self, columns: List[str], operations: List[str], query_log: Dict) -> Union[float, pd.DataFrame, str]:
        results = []
        for column in columns:
            if not self._is_valid_numeric_column(column):
                continue
            column_results = {}
            for op in operations:
                try:
                    result = self._perform_specialized_operation(column, op)
                    column_results[op] = result
                except Exception as e:
                    column_results[op] = f"Error in {op}: {str(e)}"
            results.append(column_results)
        return results[0] if len(results) == 1 else pd.DataFrame(results)

    def _is_valid_numeric_column(self, column: str) -> bool:
        return (column in self.df.columns and pd.api.types.is_numeric_dtype(self.df[column]))

    def _perform_specialized_operation(self, column: str, operation: str) -> Any:
        series = self.df[column].dropna()
        operations = {
            'sum': lambda: float(series.sum()),
            'mean': lambda: float(series.mean()),
            'median': lambda: float(series.median()),
            'std': lambda: float(series.std()),
            'min': lambda: float(series.min()),
            'max': lambda: float(series.max()),
            'range': lambda: float(series.max() - series.min()),
            'count': lambda: int(series.count()),
            'percentile': lambda: {
                '25%': float(np.percentile(series, 25)),
                '50%': float(np.percentile(series, 50)),
                '75%': float(np.percentile(series, 75))
            },
            'nth_max': lambda: float(series.nlargest(2).iloc[-1]),
            'nth_min': lambda: float(series.nsmallest(2).iloc[-1])
        }
        if operation in operations:
            return operations[operation]()
        raise ValueError(f"Unsupported operation: {operation}")

    def _generate_error_response(self, error_message: str, query_log: Dict, suggestion: str = None) -> Dict:
        error_response = {
            'error': True,
            'message': error_message,
            'query': query_log.get('original_query', 'N/A'),
            'timestamp': query_log.get('timestamp', pd.Timestamp.now())
        }
        if suggestion:
            error_response['suggestion'] = suggestion
        return error_response

    def advanced_filter_query(self, query: str) -> pd.DataFrame:
        normalized_query = self._normalize_query(query)
        conditions = self._extract_filter_conditions(normalized_query)
        filtered_df = self.df.copy()
        for condition in conditions:
            column, operator, value = condition
            if operator == '>':
                filtered_df = filtered_df[filtered_df[column] > value]
            elif operator == '<':
                filtered_df = filtered_df[filtered_df[column] < value]
            elif operator == '>=':
                filtered_df = filtered_df[filtered_df[column] >= value]
            elif operator == '<=':
                filtered_df = filtered_df[filtered_df[column] <= value]
            elif operator == '==':
                filtered_df = filtered_df[filtered_df[column] == value]
        return filtered_df

    def _extract_filter_conditions(self, query: str) -> List[Tuple[str, str, Union[int, float]]]:
        conditions = []
        comparison_patterns = [
            r'(\w+)\s*>\s*([\d.]+)',
            r'(\w+)\s*<\s*([\d.]+)',
            r'(\w+)\s*>=\s*([\d.]+)',
            r'(\w+)\s*<=\s*([\d.]+)',
            r'(\w+)\s*==\s*([\d.]+)'
        ]
        for pattern in comparison_patterns:
            matches = re.findall(pattern, query)
            conditions.extend([(match[0], pattern.split()[1], float(match[1])) for match in matches])
        doc = self.nlp(query)
        for ent in doc.ents:
            if ent.label_ in ['CARDINAL', 'QUANTITY']:
                context_window = [token.text for token in ent.lefts]
                matching_columns = self.advanced_column_matching(' '.join(context_window))
                if matching_columns:
                    conditions.append((matching_columns[0], '==', float(ent.text)))
        return conditions