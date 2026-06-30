from visualization import plot_confusion_matrix

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

from utils import (
    save_plot,
    save_metrics,
    save_results_csv
)

def evaluate_model(model,X_test , y_test , model_name):

	y_pred = model.predict(X_test)
	
	accuracy = accuracy_score(y_test,y_pred)
	precision = precision_score(y_test,y_pred)
	recall = recall_score(y_test,y_pred)
	f1 = f1_score(y_test,y_pred)
	
	report = classification_report(y_test,y_pred)
	
	metrics_text = f"""
			Model : {model_name}

			Accuracy : {accuracy:.4f}
			Precision : {precision:.4f}
			Recall : {recall:.4f}
			F1 Score : {f1:.4f}

			Classification Report
			{report}
			"""
	print(metrics_text)
	save_metrics(model_name, metrics_text)
	
	save_results_csv({
        "Model": model_name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1
    	})
	
	plot_confusion_matrix(
    				model,
    				X_test,
    				y_test,
    				model_name
			)
	
	return accuracy
