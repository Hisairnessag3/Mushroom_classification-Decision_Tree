import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from subprocess import call
import graphviz

df = pd.read_csv('mushroom2.csv')


enc = LabelEncoder()

#defining attributes
df = df[['class','cap-shape','cap-surface','cap-color','bruises','odor','gill-attachment','gill-spacing','gill-size','gill-color','stalk-shape','stalk-root','stalk-surface-above-ring','stalk-surface-below-ring','stalk-color-above-ring','stalk-color-below-ring','veil-type','veil-color','ring-number','ring-type','spore-print-color','population','habitat']]

df = df.dropna()

X = df.drop('class',axis=1)
Y = df['class']

X_train,X_test, Y_train, Y_test = train_test_split(X,Y,random_state=1)

#building classifier
model = tree.DecisionTreeClassifier()
model.fit(X_train,Y_train)
Y_predict = model.predict(X_test)

print pd.DataFrame(
    confusion_matrix(Y_test,Y_predict),
             columns=['Predicted Not Survival', 'Predicted Survival'],
             index=['True Not Survival','True Survival']
             )

tree.export_graphviz(model, out_file='tree.dot', feature_names=X.columns)

call(['dot', '-T', 'png', 'tree.dot', '-o', 'tree.png'])

