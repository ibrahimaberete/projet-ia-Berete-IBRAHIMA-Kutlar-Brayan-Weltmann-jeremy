Pour l'éxécution du projet : 

- Tout d'abord télécharger toute les dépendances à l'aide de la commande suivante : 
pip install -r requirements.txt

- Ensuite lancer la commande suivante pour exécuté FASTAPI : 
uvicorn api:app --reload 

- En parallèle éxécuter la commande suivante pour l'affichage front : 
streamlit run app.py

Pour tester le bon fonctionnement de notre FASTAPI, il faut tout d'abord exécuter le test d'entrainement.
Par la suite, il faut réaliser le test suivant pour la prediction avec les entrées suivantes : 
[
  [
    0,2,4,4,5,6,7,5,5
  ]
]