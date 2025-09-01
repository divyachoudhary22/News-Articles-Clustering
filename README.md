# News Article Clustering

Implemented an end-to-end unsupervised NLP pipeline in Python that reads news articles from category subfolders, performs cleaning (lowercasing, punctuation removal, stopword removal, optional lemmatization), and vectorizes text using TF-IDF (max_features=5000).
Selected the optimal cluster count using Silhouette analysis and the Elbow method, then trained K-Means (final model chosen from the analyzed k range) to group similar articles without labels.
Extracted and reported interpretable cluster summaries by listing top TF-IDF keywords per cluster and visualized results with TruncatedSVD + t-SNE to validate cluster separation and structure.
Quantitatively evaluated clustering quality against folder labels (used as ground truth) achieving strong agreement: ARI = 0.6812, NMI = 0.7335, Homogeneity = 0.7159, Completeness = 0.7520, V-measure = 0.7335.
Produced reproducible deliverables (cleaned dataset and clustered_articles.csv, evaluation metrics, and visualization plots); tech stack included Python, pandas, scikit-learn, NLTK, TruncatedSVD, t-SNE, and matplotlib, with code organized for easy extension (e.g., embedding-based methods or live API ingestion).
