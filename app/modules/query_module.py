import pandas as pd
import spacy
from typing import Union, Dict, List
import re
from fuzzywuzzy import process
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AdvancedDataQueryModule:
    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.logger = logging.getLogger(__name__)
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
            'top': ['highest', 'best', 'leading', 'maximum'],
            'bottom': ['lowest', 'worst', 'minimum'],
            'group': ['segment', 'category', 'cluster'],
            'count': ['total', 'number', 'quantity'],
            'average': ['mean', 'avg', 'typical']
        }
        variations = [column_name.lower()]
        for key, synonyms in semantic_map.items():
            if key in column_name.lower():
                variations.extend(synonyms)
        return list(set(variations))

    def execute_query(self, query: str) -> Union[pd.DataFrame, Dict]:
        try:
            query_log = {'original_query': query, 'timestamp': pd.Timestamp.now()}
            normalized_query = self._normalize_query(query)
            query_type = self._detect_query_type(normalized_query)
            if query_type == 'top':
                return self._handle_top_query(normalized_query)
            elif query_type == 'extreme':
                return self._handle_extreme_value_query(normalized_query)
            elif query_type == 'group':
                return self._handle_groupby_query(normalized_query)
            else:
                return self._generate_error_response("Query type not recognized", query_log, 
                                                    "Try using more specific keywords like 'top', 'maximum/minimum', or 'group by'")
        except Exception as e:
            self.logger.error(f"Error in query execution: {str(e)}")
            return self._generate_error_response(str(e), query_log)

    def _normalize_query(self, query: str) -> str:
        normalized = query.lower().strip()
        normalized = re.sub(r'\s+', ' ', normalized)
        replacements = {'show me': '', 'display': '', 'find': '', 'get': ''}
        for old, new in replacements.items():
            normalized = normalized.replace(old, new)
        return normalized.strip()

    def _detect_query_type(self, query: str) -> str:
        if any(keyword in query for keyword in ['top', 'highest', 'best', 'leading']):
            return 'top'
        elif any(keyword in query for keyword in ['maximum', 'minimum', 'highest', 'lowest']):
            return 'extreme'
        elif any(keyword in query for keyword in ['group by', 'group', 'segment', 'categorize']):
            return 'group'
        return 'unknown'

    def _handle_top_query(self, query: str) -> pd.DataFrame:
        n_match = re.search(r'top\s+(\d+)', query)
        n = int(n_match.group(1)) if n_match else 5
        col_name = self._extract_column_name(query)
        condition_value = self._extract_numeric_value(query)
        if condition_value is not None:
            filtered_df = self.df[self.df[col_name] > condition_value]
        else:
            filtered_df = self.df
        return filtered_df.nlargest(n, col_name)

    def _handle_extreme_value_query(self, query: str) -> pd.DataFrame:
        col_name = self._extract_column_name(query)
        is_max = any(word in query for word in ['maximum', 'highest'])
        if is_max:
            return self.df[self.df[col_name] == self.df[col_name].max()]
        else:
            return self.df[self.df[col_name] == self.df[col_name].min()]

    def _handle_groupby_query(self, query: str) -> pd.DataFrame:
        group_col = self._extract_group_column(query)
        agg_col = self._extract_column_name(query, exclude=[group_col])
        agg_funcs = self._determine_aggregations(query)
        return self.df.groupby(group_col)[agg_col].agg(agg_funcs).reset_index()

    def _extract_column_name(self, query: str, exclude: List[str] = None) -> str:
        exclude = exclude or []
        doc = self.nlp(query)
        for col in self.df.columns:
            if col.lower() in query and col not in exclude:
                return col
        valid_columns = [col for col in self.df.columns if col not in exclude]
        matches = process.extract(query, valid_columns, limit=1)
        if matches and matches[0][1] > 70:
            return matches[0][0]
        raise ValueError("Column name not found in query")

    def _extract_numeric_value(self, query: str) -> Union[float, None]:
        matches = re.findall(r'>\s*([\d.]+)', query)
        return float(matches[0]) if matches else None

    def _extract_group_column(self, query: str) -> str:
        group_patterns = [r'group by\s+(\w+)', r'grouped by\s+(\w+)', r'segment by\s+(\w+)']
        for pattern in group_patterns:
            match = re.search(pattern, query)
            if match:
                column = match.group(1)
                if column in self.df.columns:
                    return column
        raise ValueError("Group column not found in query")

    def _determine_aggregations(self, query: str) -> List[str]:
        agg_funcs = ['mean', 'count']
        if 'sum' in query or 'total' in query:
            agg_funcs.append('sum')
        if 'maximum' in query or 'highest' in query:
            agg_funcs.append('max')
        if 'minimum' in query or 'lowest' in query:
            agg_funcs.append('min')
        return list(set(agg_funcs))

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