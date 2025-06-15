import pandas as pd
import spacy
from typing import Union, Dict, Any, List
import re
from fuzzywuzzy import process
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import logging
import numpy as np
from app.modules.math_module import UltraAdvancedMathModule
from app.modules.query_module import AdvancedDataQueryModule

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class OpenQAModule:
    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.math_module = UltraAdvancedMathModule(df)
        self.data_query_module = AdvancedDataQueryModule(df)
        self.model_name = "facebook/opt-350m"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)

    def process_query(self, query: str) -> Union[str, pd.DataFrame, Dict[str, Any]]:
        try:
            normalized_query = query.lower().strip()
            if any(term in normalized_query for term in ['calculate', 'sum', 'average', 'mean', 'median', 'standard deviation', 'min', 'max', 'range']):
                return self.math_module.execute_math_query(query)
            elif any(term in normalized_query for term in ['top', 'bottom', 'maximum', 'minimum', 'extreme', 'group by', 'filter']):
                return self.data_query_module.execute_query(query)
            elif 'where' in normalized_query or 'condition' in normalized_query:
                return self.math_module.advanced_filter_query(query)
            else:
                context = self._generate_context()
                return self._generate_answer(query, context)
        except Exception as e:
            return {
                'error': True,
                'message': f"Query processing error: {str(e)}",
                'suggestion': "Try rephrasing your query or being more specific"
            }

    def _generate_context(self) -> str:
        context_parts = [
            f"Dataset contains {len(self.df)} rows and {len(self.df.columns)} columns.",
            "Columns: " + ", ".join(self.df.columns),
            "Column types: " + ", ".join([f"{col}: {str(self.df[col].dtype)}" for col in self.df.columns])
        ]
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            context_parts.append(
                f"{col} summary: Mean = {self.df[col].mean():.2f}, Min = {self.df[col].min()}, Max = {self.df[col].max()}"
            )
        return " ".join(context_parts)

    def _generate_answer(self, query: str, context: str) -> str:
        try:
            prompt = f"""Context: {context}
Question: {query}
Provide a concise and informative answer based on the context:"""
            inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(self.device)
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=150,
                num_return_sequences=1,
                temperature=0.7,
                do_sample=True
            )
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            return response.split("Provide a concise")[0].strip()
        except Exception as e:
            return f"Error generating answer: {str(e)}"

    def get_advanced_insights(self) -> Dict[str, Any]:
        insights = {
            'basic_info': {'rows': len(self.df), 'columns': list(self.df.columns)},
            'numeric_summary': {},
            'categorical_summary': {}
        }
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            insights['numeric_summary'][col] = {
                'mean': self.df[col].mean(),
                'median': self.df[col].median(),
                'min': self.df[col].min(),
                'max': self.df[col].max(),
                'std': self.df[col].std()
            }
        categorical_cols = self.df.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            insights['categorical_summary'][col] = {
                'unique_values': self.df[col].nunique(),
                'top_values': self.df[col].value_counts().nlargest(5).to_dict()
            }
        return insights