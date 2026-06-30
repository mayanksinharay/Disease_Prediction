import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve
from sklearn.metrics import auc

from sklearn.metrics import confusion_matrix
from utils import save_plot

def plot_confusion_matrix(model, X_test, y_test, model_name):
	y_pred = model.predict(X_test)
	cm = confusion_matrix(y_test, y_pred)
	plt.figure(figsize=(6,5))
	
	sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues"
    	)

	plt.title(f"{model_name} Confusion Matrix")
	plt.xlabel("Predicted")
	plt.ylabel("Actual")
	
	save_plot(
        f"{model_name}_confusion_matrix.png",
        show=False
    	)

def plot_roc_curve(model, X_test, y_test, model_name):
	y_score = model.predict_proba(X_test)[:,1]
	fpr, tpr, _ = roc_curve(y_test, y_score)
	roc_auc = auc(fpr, tpr)
	plt.plot(
        fpr,
        tpr,
        linewidth=2,
        label=f"{model_name} (AUC = {roc_auc:.3f})"
    	)
	
	return roc_auc