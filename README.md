# Métodos-Ensemble
Machine Learning Metodos Ensemble

## 1.- Bagging Classifier
Bagging é usado para construção de múltiplos modelos (normalmente do mesmo tipo) a partir de diferentes subsets no dataset de treino.
Um classificador Bagging é um meta-estimador ensemble que faz o fit de classificadores base, cada um em subconjuntos aleatórios do conjunto de dados original e, em seguida, agrega suas previsões individuais (por votação ou por média) para formar uma previsão final.

Tal meta-estimador pode tipicamente ser usado como uma maneira de reduzir a variância de um estimador (por exemplo, uma árvore de decisão), introduzindo a randomização em seu procedimento de construção e fazendo um ensemble (conjunto) a partir dele.

A Bagging classifier is an ensemble meta-estimator that fits base classifiers each on random subsets of the original dataset and then aggregate their individual predictions (either by voting or by averaging) to form a final prediction. Such a meta-estimator can typically be used as a way to reduce the variance of a black-box estimator (e.g., a decision tree), by introducing randomization into its construction procedure and then making an ensemble out of it.

This algorithm encompasses several works from the literature. When random subsets of the dataset are drawn as random subsets of the samples, then this algorithm is known as Pasting [1]. If samples are drawn with replacement, then the method is known as Bagging [2]. When random subsets of the dataset are drawn as random subsets of the features, then the method is known as Random Subspaces [3]. Finally, when base estimators are built on subsets of both samples and features, then the method is known as Random Patches
![image](https://user-images.githubusercontent.com/87387315/165098797-255cd282-c3e3-40fb-8db0-dacb678fdddb.png)

  a.-Os conjuntos de dados de treinamento consistem em amostras de dados representadas usando cores diferentes.
  
  b.-Amostras aleatórias são retiradas com reposição. Isso significa essencialmente que pode haver dados duplicados em cada uma das amostras.
  
  c.-Cada amostra é usada para treinar diferentes estimadores (regressores) / classificadores representados usando classificador 1, classificador 2, …, classificador n
  
  d.-É criado um classificador/regressor ensemble que pega as previsões de diferentes classificadores/regressores e faz a previsão final com base na votação ou na média, respectivamente.
  
  e.-O desempenho do classificador ensemble é testado usando o conjunto de dados de treinamento.

### Parameters : 
base_estimator : default = None
O estimador de base para treinar em subconjuntos aleatórios do conjunto de dados. Em caso de não ser especificado None, o estimador base é o DecisionTreeClassifier.

The base estimator to fit on random subsets of the dataset. If None, then the base estimator is a DecisionTreeClassifier.

n_estimators: int, default=10
Número de estimadores base no emsemble o default é 10.
The number of base estimators in the ensemble.

max_samples: int or float, default=1.0
O número de amostras a serem extraídas de X para treinar cada estimador de base
The number of samples to draw from X to train each base estimator

Outros parámetros:

http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.BaggingClassifier.html

### Beneficios do Bagging? / Bagging classifier Help
Ajuda a **reduzir a variância de estimadores individuais**, introduzindo a randomização no estágio de treinamento de cada um dos estimadores e fazendo um conjunto de todos os estimadores.

Helps reducing the variance of individual estimators by introducing randomization into the training stage of each of the estimators and making an ensemble out of all the estimators.

### Quando usar Bagging Classifier? / When to use Bagging Classifier?
Bagging Classifier é usado em modelos onde existe uma alta variância e baixo vies, conseguindo um alto beneficio em modelos de árvores de decisão, modelos lineares de baixa variância podem não se beneficiar muito com o uso de esta técnica. 

Bagging classifier helps reduce the variance of unstable classifiers (having high variance). The unstable classifiers include classifiers trained using algorithms such as decision tree which is found to have high variance and low bias.

Variancia e vies : https://vitalflux.com/bias-variance-concepts-interview-questions/

![image](https://user-images.githubusercontent.com/87387315/165101994-92be56ac-d761-422f-8375-615cb3b8e982.png)

### Example




