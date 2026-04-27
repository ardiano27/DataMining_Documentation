import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from pymongo import MongoClient
from pymongo.errors import ServerSelectionTimeoutError


MONGO_URI = "mongodb://localhost:27017/"
DB_NAME = "data_mining"
COLLECTION_NAME = "iris"


def load_iris_dataframe():
    iris = load_iris()
    df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
    df["target"] = iris.target
    df["target_name"] = df["target"].map({
        0: "setosa",
        1: "versicolor",
        2: "virginica",
    })
    return df, iris.feature_names


def connect_mongo(uri=MONGO_URI):
    client = MongoClient(uri, serverSelectionTimeoutMS=5000)
    client.admin.command("ping")
    return client


def save_dataframe_to_mongo(df, client):
    db = client[DB_NAME]
    collection = db[COLLECTION_NAME]

    # Supaya tidak dobel setiap kali script dijalankan ulang.
    collection.delete_many({})

    records = df.to_dict("records")
    result = collection.insert_many(records)
    print(f"Data berhasil disimpan ke MongoDB: {len(result.inserted_ids)} dokumen")
    return collection


def load_dataframe_from_mongo(collection):
    records = list(collection.find({}, {"_id": 0}))
    return pd.DataFrame(records)


def train_knn(df, feature_cols):
    X = df[feature_cols]
    y = df["target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = KNeighborsClassifier(n_neighbors=5)
    model.fit(X_train_scaled, y_train)

    y_pred = model.predict(X_test_scaled)

    print("\nAkurasi:", round(accuracy_score(y_test, y_pred), 4))
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))


def main():
    df, feature_cols = load_iris_dataframe()
    print("Data iris dari sklearn:")
    print(df.head())

    try:
        client = connect_mongo()
    except ServerSelectionTimeoutError:
        print("Gagal konek ke MongoDB. Pastikan MongoDB lokal sedang berjalan di port 27017.")
        return

    collection = save_dataframe_to_mongo(df, client)
    df_mongo = load_dataframe_from_mongo(collection)

    print("\nData yang dibaca kembali dari MongoDB:")
    print(df_mongo.head())

    train_knn(df_mongo, feature_cols)


if __name__ == "__main__":
    main()
