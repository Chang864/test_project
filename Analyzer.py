"""Analysis"""
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import roc_curve, auc, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns
import os


class DeviceTestAnalyzer:
    """Test Analyzer"""
    
    def __init__(self, data: pd.DataFrame):
        self.data = data
        self.results = {}
    
    def run_full_analysis(self, output_dir: str = 'results'):
        """"""
        os.makedirs(output_dir, exist_ok=True)
        
        print("1. Exploratory analysis")
        self.exploratory_analysis(output_dir)
        
        print("2. Hypothesis test")
        self.hypothesis_test(output_dir)  
        
        print("3. Train models")
        self.train_prediction_model(output_dir)
        
        print("4. Generate report")
        self.generate_report(output_dir)
    
    def exploratory_analysis(self, output_dir: str = None): 
        """exploratory data analysis"""
        # statistical analysis
        stats = {
            'total_tests': len(self.data),
            'devices': self.data['name'].nunique(),
            'failure_rate': (self.data['test_status'] == 'fail').mean(),
            'avg_angle_x': self.data['angle_x'].mean(),
            'avg_angle_y': self.data['angle_y'].mean(),
            'avg_angle_z': self.data['angle_z'].mean(),
        }
        
        # visualizatiion
        if output_dir:
            try:
                import matplotlib.pyplot as plt
                
                fig, axes = plt.subplots(1, 2, figsize=(12, 5))
                
                # test result distribution
                status_counts = self.data['test_status'].value_counts()
                axes[0].pie(status_counts.values, labels=status_counts.index, autopct='%1.1f%%')
                axes[0].set_title('Test Results Distribution')
                
                # angle distribution
                self.data[['angle_x', 'angle_y', 'angle_z']].boxplot(ax=axes[1])
                axes[1].set_title('Angle Distributions')
                
                plt.tight_layout()
                plt.savefig(f'{output_dir}/exploratory_plots.png', dpi=150)
                plt.close()
                print(f"Exploratory plots saved to: {output_dir}/")
            except ImportError:
                print("matplotlib not available, skipping plots")
            except Exception as e:
                print(f"Could not save plots: {e}")
        
        self.results['stats'] = stats
        return stats
    
    # Hypothesis testing
    def hypothesis_test(self, output_dir: str = None):
        """
        Test fail rate against mark (up and down)
        
        Returns:
            Test results
        """
        print("Hypothesis test: mark and fail rate")
        
        # check data
        if 'mark' not in self.data.columns or 'test_status' not in self.data.columns:
            print("Missing columns: mark or test_status")
            return None
        
        # Different groups
        up_data = self.data[self.data['mark'] == 'up']
        down_data = self.data[self.data['mark'] == 'down']
        
        n_up = len(up_data)
        n_down = len(down_data)
        
        if n_up == 0 or n_down == 0:
            print("One group is empty, no test.")
            return None
        
        failures_up = (up_data['test_status'] == 'fail').sum()
        failures_down = (down_data['test_status'] == 'fail').sum()
    
        rate_up = failures_up / n_up
        rate_down = failures_down / n_down
        
        print(f"Data summary:")
        print(f"  group up: {n_up} samples, {failures_up} failed, fail rate: {rate_up:.2%}")
        print(f"  group down: {n_down} samples, {failures_down} failed, fail rate: {rate_down:.2%}")
        print(f"  difference: {abs(rate_up - rate_down):.2%}")
        
        # z test
        from statsmodels.stats.proportion import proportions_ztest
        
        count = [failures_up, failures_down]
        nobs = [n_up, n_down]
        
        try:
            z_stat, p_value = proportions_ztest(count, nobs, alternative='two-sided')
            
            result = {
                'sample_sizes': {'up': n_up, 'down': n_down},
                'failure_counts': {'up': int(failures_up), 'down': int(failures_down)},
                'failure_rates': {'up': float(rate_up), 'down': float(rate_down)},
                'z_test': {
                    'z_statistic': float(z_stat),
                    'p_value': float(p_value),
                    'is_significant': p_value < 0.05
                }
            }
            
            print(f"Test results:")
            print(f"  Z: {z_stat:.4f}")
            print(f"  P-value: {p_value:.4f}")
            print(f"  Significant(p<0.05): {'Yes' if p_value < 0.05 else 'No'}")
            
            # Conclusions
            if p_value < 0.05:
                if rate_up > rate_down:
                    print(f"Group up has a significantly higher failure rate")
                else:
                    print(f"Group down has a significantly higher failure rate")
            else:
                print(f"No significant difference between 2 groups")
            
            # Save results
            if output_dir and output_dir.strip():
                import os
                import json
                
                os.makedirs(output_dir, exist_ok=True)
                with open(f'{output_dir}/hypothesis_test.json', 'w') as f:
                    json.dump(result, f, indent=2)
                print(f"Test result saved at: {output_dir}/")
            
            return result
            
        except Exception as e:
            print(f"Test failed: {e}")
            return None
        
    def _save_model_results(self, model, X_test, y_test, y_pred, output_dir):
        """
        Save model results to file
        """
        import os
        import pandas as pd
        import json
        
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"Saving model results to: {output_dir}")
        
        # 1. Feature importance
        if hasattr(model, 'feature_importances_'):
            importance_df = pd.DataFrame({
                'feature': X_test.columns,
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            importance_df.to_csv(f'{output_dir}/feature_importance.csv', index=False)
            print(f" Feature importance: {output_dir}/feature_importance.csv")
        
        # 2. Predicted results
        results_df = pd.DataFrame({
            'y_true': y_test.values,
            'y_pred': y_pred,
            'is_correct': y_test.values == y_pred
        })
        results_df.to_csv(f'{output_dir}/predictions.csv', index=False)
        print(f" Predictions: {output_dir}/predictions.csv")
        
        # 3. Classification report
        from sklearn.metrics import classification_report
        report = classification_report(y_test, y_pred, 
                                      target_names=['pass', 'fail'], 
                                      output_dict=True)
        
        with open(f'{output_dir}/classification_report.json', 'w') as f:
            json.dump(report, f, indent=2)
        print(f" Classification report: {output_dir}/classification_report.json")
        
        # 4. ROC curve
        try:
            y_prob = model.predict_proba(X_test)[:, 1]
            from sklearn.metrics import roc_curve, auc
            
            fpr, tpr, thresholds = roc_curve(y_test, y_prob)
            roc_data = {
                'fpr': fpr.tolist(),
                'tpr': tpr.tolist(),
                'auc': auc(fpr, tpr)
            }
            
            with open(f'{output_dir}/roc_data.json', 'w') as f:
                json.dump(roc_data, f, indent=2)
            print(f"  ROC data: {output_dir}/roc_data.json")
            
        except Exception as e:
            print(f" Could not save ROC data: {e}")
        
        # 5. Model summary
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        
        metrics = {
            'accuracy': float(accuracy_score(y_test, y_pred)),
            'precision': float(precision_score(y_test, y_pred, zero_division=0)),
            'recall': float(recall_score(y_test, y_pred, zero_division=0)),
            'f1_score': float(f1_score(y_test, y_pred, zero_division=0)),
            'test_samples': int(len(y_test)),
            'positive_samples': int(y_test.sum()),
            'predicted_positive': int(y_pred.sum())
        }
        
        with open(f'{output_dir}/model_summary.txt', 'w') as f:
            f.write("Model Performance Summary\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Accuracy: {metrics['accuracy']:.4f}\n")
            f.write(f"Precision: {metrics['precision']:.4f}\n")
            f.write(f"Recall: {metrics['recall']:.4f}\n")
            f.write(f"F1 Score: {metrics['f1_score']:.4f}\n\n")
            f.write(f"Test samples: {metrics['test_samples']}\n")
            f.write(f"Actual failures: {metrics['positive_samples']}\n")
            f.write(f"Predicted failures: {metrics['predicted_positive']}\n\n")
            
            if hasattr(model, 'feature_importances_'):
                f.write("Feature Importance:\n")
                for idx, row in importance_df.iterrows():
                    f.write(f"  {row['feature']}: {row['importance']:.4f}\n")
        
        print(f" Model summary: {output_dir}/model_summary.txt")
    
    def train_prediction_model(self, output_dir: str = None):
        print(f"Start to train model output_dir={output_dir}")
        """Model to predict whether the test will fail"""
        # Prepare variables
        X = self.data[['angle_x', 'angle_y', 'angle_z','number_of_drop']].copy()
        X['mark'] = (self.data['mark'] == 'up').astype(int)
        
        # Prepare target variable
        y = (self.data['test_status'] == 'fail').astype(int)
        
        # Train test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.4, random_state=101, stratify=y
        )
        
        # Train model
        model = RandomForestClassifier(n_estimators=50, random_state=101, class_weight='balanced')
        model.fit(X_train, y_train)
        
        # Model evaluation
        y_pred = model.predict(X_test)
        
        ac_score = accuracy_score(y_test, y_pred)
        predict_probs = model.predict_proba(X_test)[:, 1]
        fpr, tpr, thr = roc_curve(y_test, predict_probs)
        m_auc = auc(fpr, tpr)
        
        # Feature importance for RF
        importance = pd.DataFrame({
            'feature': X.columns,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        # Get results
        report = classification_report(y_test, y_pred, 
                                       target_names=['pass', 'fail'],
                                       output_dict=True)
        
        self.results['model_performance'] = report
        self.results['feature_importance'] = importance
        
        if output_dir and output_dir.strip():
            print(f"Results saved at: {output_dir}")
            self._save_model_results(model, X_test, y_test, y_pred, output_dir)
        else:
            print("Return Results without saving to a file")
            print(f"Model accuracy: {report['accuracy']:.2%}")
            
            print('AUC:', m_auc)
            print(f'Confusion matrix: {confusion_matrix(y_test, y_pred)}')
            
            print(f"The most important feature: {importance.iloc[0]['feature']}")
            print(f'Feature importance: {importance.to_string()}')
            print(f"Report: {classification_report(y_test, y_pred, target_names=['pass', 'fail'])}")
    
        return model
    
    def generate_report(self, output_dir: str):
        """Generate report"""
        report_lines = [
            "Drop test analysis",
            "=" * 50,
            "",
            "1. Summary",
            f"   Total number: {self.results['stats']['total_tests']}",
            f"   Number of devices: {self.results['stats']['devices']}",
            f"   Failure Rate: {self.results['stats']['failure_rate']:.2%}",
            "",
            "2. Model Performance",
            f"   Accuracy: {self.results['model_performance']['accuracy']:.2%}",
            f"   Most important feature: {self.results['feature_importance'].iloc[0]['feature']}",
            "",
            "3. Recommendations",
            "   1. Monitor abnormal angle values, especially angle_z",
            "   2. Calibrate devices with high failure rates",
            "   3. Optimize test procedures to improve stability",
        ]
        
        report_path = f'{output_dir}/analysis_report.txt'
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(report_lines))
        
        print(f" Report generated: {report_path}")
