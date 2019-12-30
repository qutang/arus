"""
Plot confusion matrix
=====================
"""

from arus.models.muss import MUSSModel
import matplotlib.pyplot as plt
muss = MUSSModel()
input_class = ['Sit', 'Stand', 'Stand', 'Walk']
predict_class = ['Sit', 'Walk', 'Stand', 'Stand']
fig = muss.get_confusion_matrix(input_class, predict_class, labels=[
    'Sit', 'Stand', 'Walk'], graph=True)

df = muss.get_confusion_matrix(input_class, predict_class, labels=[
    'Sit', 'Stand', 'Walk'], graph=False)

print(df)
