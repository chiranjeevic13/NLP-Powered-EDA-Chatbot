import pandas as pd
import numpy as np
import os
import logging
from datetime import datetime
import json
import yaml
import csv
from scipy import stats

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EDAModule:
    def __init__(self, df: pd.DataFrame, output_dir='eda_reports'):
        self.df = df
        os.makedirs(output_dir, exist_ok=True)
        self.output_dir = output_dir
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.numeric_cols = df.select_dtypes(include=[np.number]).columns
        self.categorical_cols = df.select_dtypes(include=['object']).columns
        self.insights = {}

    def generate_full_report(self) -> str:
        """Generate comprehensive EDA report with advanced insights."""
        self._check_scalability()
        self._advanced_statistical_analysis()
        self._analyze_distributions()
        self._detect_outliers()
        self._recommend_feature_engineering()
        self._analyze_correlations()
        return self._format_insights()

    def _check_scalability(self):
        """Perform scalability and performance checks."""
        self.insights['scalability'] = {
            'total_memory_usage': float(self.df.memory_usage(deep=True).sum() / 1024**2),
            'row_count': int(len(self.df)),
            'column_count': int(len(self.df.columns)),
            'sparsity_ratio': float(1 - (self.df.count().sum() / (self.df.shape[0] * self.df.shape[1])))
        }

    def _advanced_statistical_analysis(self):
        """Conduct advanced statistical analysis."""
        numeric_insights = {}
        for col in self.numeric_cols:
            numeric_insights[col] = {
                'skewness': float(self.df[col].skew()),
                'kurtosis': float(self.df[col].kurtosis()),
                'coefficient_of_variation': float(self.df[col].std() / self.df[col].mean()) if self.df[col].mean() != 0 else float('inf'),
                'quantiles': {
                    '25%': float(self.df[col].quantile(0.25)),
                    '50%': float(self.df[col].quantile(0.5)),
                    '75%': float(self.df[col].quantile(0.75))
                }
            }

        categorical_insights = {}
        for col in self.categorical_cols:
            categorical_insights[col] = {
                'unique_values_count': int(self.df[col].nunique()),
                'top_5_values': dict(self.df[col].value_counts().head()),
                'entropy': float(self._calculate_entropy(self.df[col]))
            }

        self.insights['statistical_analysis'] = {
            'numeric_columns': numeric_insights,
            'categorical_columns': categorical_insights
        }

    def _analyze_distributions(self):
        """Analyze distribution characteristics."""
        distribution_insights = {}
        for col in self.numeric_cols:
            _, p_value = stats.normaltest(self.df[col].dropna())
            distribution_insights[col] = {
                'is_normal_distribution': bool(p_value > 0.05),
                'shapiro_wilk_test_p_value': float(p_value)
            }
        self.insights['distribution_analysis'] = distribution_insights

    def _detect_outliers(self):
        """Detect outliers using IQR method."""
        outlier_insights = {}
        for col in self.numeric_cols:
            Q1 = self.df[col].quantile(0.25)
            Q3 = self.df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR

            outliers = self.df[(self.df[col] < lower_bound) | (self.df[col] > upper_bound)]
            outlier_insights[col] = {
                'outlier_count': int(len(outliers)),
                'outlier_percentage': float(len(outliers) / len(self.df) * 100),
                'lower_bound': float(lower_bound),
                'upper_bound': float(upper_bound)
            }

        self.insights['outlier_analysis'] = outlier_insights

    def _recommend_feature_engineering(self):
        """Provide feature engineering recommendations."""
        recommendations = []
        for col in self.numeric_cols:
            if self.insights['statistical_analysis']['numeric_columns'][col]['coefficient_of_variation'] > 1:
                recommendations.append(f"Consider binning column {col} due to high variability")
        for col in self.categorical_cols:
            if self.insights['statistical_analysis']['categorical_columns'][col]['unique_values_count'] > 10:
                recommendations.append(f"Consider advanced encoding for {col} (e.g., target encoding)")
        self.insights['feature_engineering_recommendations'] = recommendations

    def _analyze_correlations(self):
        """Analyze correlations between numerical columns."""
        if len(self.numeric_cols) > 1:
            correlation_matrix = self.df[self.numeric_cols].corr()
            high_correlations = []
            for i in range(len(self.numeric_cols)):
                for j in range(i + 1, len(self.numeric_cols)):
                    col1 = self.numeric_cols[i]
                    col2 = self.numeric_cols[j]
                    corr = correlation_matrix.loc[col1, col2]
                    if abs(corr) > 0.5:
                        high_correlations.append({
                            'columns': (col1, col2),
                            'correlation': float(corr)
                        })
            self.insights['correlation_analysis'] = high_correlations
        else:
            self.insights['correlation_analysis'] = "Insufficient numerical columns for correlation analysis"

    def _calculate_entropy(self, series):
        """Calculate entropy for categorical variable."""
        value_counts = series.value_counts(normalize=True)
        return -np.sum(value_counts * np.log2(value_counts))

    def save_insights(self, format='all'):
        """Save insights in multiple file formats."""
        base_filename = f"{self.output_dir}/eda_insights_{self.timestamp}"
        if format in ['json', 'all']:
            with open(f"{base_filename}.json", 'w') as f:
                json.dump(self.insights, f, indent=4)
        if format in ['yaml', 'yml', 'all']:
            with open(f"{base_filename}.yaml", 'w') as f:
                yaml.dump(self.insights, f, default_flow_style=False)
        if format in ['csv', 'all']:
            self._save_insights_to_csv(base_filename)
        if format in ['txt', 'text', 'all']:
            with open(f"{base_filename}.txt", 'w') as f:
                f.write(self._format_insights())
        return f"Insights saved in {base_filename}.*"

    def _save_insights_to_csv(self, base_filename):
        """Convert nested insights to flat CSV structure."""
        rows = []
        def flatten_dict(d, parent_key='', sep='_'):
            items = []
            for k, v in d.items():
                new_key = f"{parent_key}{sep}{k}" if parent_key else k
                if isinstance(v, dict):
                    items.extend(flatten_dict(v, new_key, sep=sep).items())
                else:
                    items.append((new_key, v))
            return dict(items)
        for section, content in self.insights.items():
            if isinstance(content, dict):
                flattened = flatten_dict({section: content})
                rows.append(flattened)
            elif isinstance(content, list):
                for item in content:
                    rows.append(flatten_dict({section: item}))
        if rows:
            keys = set().union(*[d.keys() for d in rows])
            with open(f"{base_filename}.csv", 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=sorted(keys))
                writer.writeheader()
                writer.writerows(rows)

    def _format_insights(self) -> str:
        """Format insights into a readable report."""
        report_sections = []
        scalability = self.insights['scalability']
        report_sections.append(f"""Scalability Analysis:
- Total Memory Usage: {scalability['total_memory_usage']:.2f} MB
- Total Rows: {scalability['row_count']}
- Total Columns: {scalability['column_count']}
- Data Sparsity Ratio: {scalability['sparsity_ratio']:.2%}""")
        report_sections.append("\nDetailed Statistical Analysis:")
        for col_type, columns in self.insights['statistical_analysis'].items():
            report_sections.append(f"\n{col_type.replace('_', ' ').title()}:")
            for col, stats in columns.items():
                report_sections.append(f"- {col}: {stats}")
        report_sections.append("\nDistribution Analysis:")
        for col, analysis in self.insights['distribution_analysis'].items():
            report_sections.append(f"- {col}: {'Normal' if analysis['is_normal_distribution'] else 'Non-Normal'} Distribution")
        report_sections.append("\nOutlier Analysis:")
        for col, outliers in self.insights['outlier_analysis'].items():
            report_sections.append(f"- {col}: {outliers['outlier_percentage']:.2f}% Outliers")
        report_sections.append("\nFeature Engineering Recommendations:")
        report_sections.append('\n'.join(self.insights['feature_engineering_recommendations']))
        return '\n'.join(report_sections)

    def run_and_save(self, formats=['json', 'yaml', 'txt']):
        """Run the full report generation and save insights in specified formats."""
        logger.info("Generating full EDA report...")
        self.generate_full_report()
        logger.info("Saving insights in formats: %s", formats)
        for fmt in formats:
            self.save_insights(format=fmt)
        logger.info("EDA report generation and saving completed.")