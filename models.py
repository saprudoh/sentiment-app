import re
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from dataset import get_test_data
from sklearn.metrics import accuracy_score

class SentimentAnalyzer:
    def __init__(self):
        # --- Dummy Dataset ---
        # Dataset ini digunakan untuk melatih model secara "on-the-fly".
        # Di aplikasi nyata, Anda akan memiliki dataset yang lebih besar dan disimpan terpisah.
        self.dummy_data = {
            'comment': [
                # Positif
                "Keren banget videonya!", "Sangat bermanfaat.", "Aku suka sekali.",
                "Penjelasannya sangat jelas dan mudah dipahami.", "Luar biasa, saya belajar banyak.",
                "Terima kasih atas informasinya, sangat membantu.", "Kualitas kontennya sangat tinggi.",
                "Saya sangat merekomendasikan channel ini.", "Isinya daging semua, tidak ada basa-basi.",
                "Pembicaranya asik dan tidak membosankan.", "Ini adalah video terbaik yang pernah saya lihat.",
                "Sangat inspiratif dan memotivasi.", "Editing videonya profesional.",
                "Suaranya jernih dan enak didengar.", "Materinya relevan dengan yang saya cari.",

                # Negatif
                "Tidak jelas.", "Membosankan.", "Konten sampah.",
                "Saya tidak mengerti sama sekali.", "Kualitas videonya buruk, buram.",
                "Suaranya kecil sekali, tidak terdengar.", "Terlalu banyak iklan, mengganggu.",
                "Pembicaranya berbelit-belit, tidak langsung ke inti.", "Ini sama sekali tidak membantu.",
                "Informasi yang diberikan salah.", "Judulnya clickbait, isinya tidak sesuai.",
                "Saya kecewa dengan konten ini.", "Buang-buang waktu saja menonton ini.",
                "Materi yang disampaikan sudah usang.", "Terlalu banyak teori, kurang praktek.",

                # Netral
                "Biasa saja.", "Cukup informatif.", "Lumayan lah.",
                "Video ini membahas tentang sejarah.", "Durasi video ini adalah 10 menit.",
                "Saya akan mencoba mempraktekkannya nanti.", "Ini adalah sebuah ulasan produk.",
                "Oke, saya mengerti sekarang.", "Rapat akan diadakan besok.",
                "Harap perhatikan pengumuman berikutnya.", "File dapat diunduh pada link di deskripsi.",
                "Ini adalah bagian pertama dari seri video.", "Jadwal akan diperbarui setiap minggu.",
                "Silakan tinggalkan komentar di bawah.", "Video ini direkam menggunakan kamera DSLR."
            ],
            'sentiment': [
                # Positif
                'positif', 'positif', 'positif', 'positif', 'positif',
                'positif', 'positif', 'positif', 'positif', 'positif',
                'positif', 'positif', 'positif', 'positif', 'positif',
                # Negatif
                'negatif', 'negatif', 'negatif', 'negatif', 'negatif',
                'negatif', 'negatif', 'negatif', 'negatif', 'negatif',
                'negatif', 'negatif', 'negatif', 'negatif', 'negatif',
                # Netral
                'netral', 'netral', 'netral', 'netral', 'netral',
                'netral', 'netral', 'netral', 'netral', 'netral',
                'netral', 'netral', 'netral', 'netral', 'netral'
            ]
        }
        self.df_train = pd.DataFrame(self.dummy_data)
        
        # Inisialisasi model
        self.vectorizer = TfidfVectorizer(preprocessor=self._preprocess_text)
        self.svm_model = SVC(kernel='linear')
        self.nb_model = MultinomialNB()
        self.vader_analyzer = SentimentIntensityAnalyzer()

        # Membuat pipeline untuk setiap algoritma
        self.svm_pipeline = self._create_pipeline(self.svm_model)
        self.nb_pipeline = self._create_pipeline(self.nb_model)
        
        # Latih model saat inisialisasi
        self._train()

    def _preprocess_text(self, text):
        """Fungsi untuk membersihkan teks."""
        text = text.lower()  # Ubah ke huruf kecil
        text = re.sub(r'\d+', '', text)  # Hapus angka
        text = re.sub(r'[^\w\s]', '', text)  # Hapus tanda baca
        text = text.strip()  # Hapus spasi di awal dan akhir
        return text

    def _create_pipeline(self, model):
        """Membuat pipeline scikit-learn."""
        return Pipeline([
            ('vectorizer', self.vectorizer),
            ('classifier', model)
        ])

    def _train(self):
        """Melatih model SVM dan Naive Bayes."""
        X_train = self.df_train['comment']
        y_train = self.df_train['sentiment']

        # Melatih pipeline
        self.svm_pipeline.fit(X_train, y_train)
        self.nb_pipeline.fit(X_train, y_train)

    def analyze(self, comments_to_analyze, algorithm='svm'):
        """
        Menganalisis sentimen dari daftar komentar.
        
        Args:
            comments_to_analyze (list): Daftar string komentar yang akan dianalisis.
            algorithm (str): 'svm', 'naive_bayes', atau 'lexicon'.
            
        Returns:
            list: Daftar dictionary berisi 'Komentar' dan 'Sentimen'.
        """
        results = []

        if algorithm == 'lexicon':
            for comment in comments_to_analyze:
                score = self.vader_analyzer.polarity_scores(comment)
                sentiment = 'Netral'
                if score['compound'] >= 0.05:
                    sentiment = 'Positif'
                elif score['compound'] <= -0.05:
                    sentiment = 'Negatif'
                results.append({'Komentar': comment, 'Sentimen': sentiment})
            return results

        # Pilih model yang sudah dilatih
        if algorithm == 'svm':
            model = self.svm_pipeline
        elif algorithm == 'naive_bayes':
            model = self.nb_pipeline
        else:
            raise ValueError("Algoritma tidak valid. Pilih 'svm', 'naive_bayes', atau 'lexicon'.")

        # Lakukan prediksi
        predictions = model.predict(comments_to_analyze)
        
        # Gabungkan komentar dengan hasil prediksi
        for comment, sentiment in zip(comments_to_analyze, predictions):
            results.append({'Komentar': comment, 'Sentimen': sentiment.capitalize()})
            
        return results

    def calculate_accuracy(self, algorithm='svm'):
        """
        Menghitung akurasi model terhadap dataset pengujian.

        Args:
            algorithm (str): 'svm', 'naive_bayes', atau 'lexicon'.

        Returns:
            float: Nilai akurasi dalam persentase.
        """
        test_data = get_test_data()
        comments, true_labels = zip(*test_data)
        
        # Normalisasi label asli (menjadi huruf kecil)
        true_labels = [label.lower() for label in true_labels]

        predicted_labels = []

        if algorithm == 'lexicon':
            for comment in comments:
                score = self.vader_analyzer.polarity_scores(comment)
                sentiment = 'neutral'
                if score['compound'] >= 0.05:
                    sentiment = 'positive'
                elif score['compound'] <= -0.05:
                    sentiment = 'negative'
                predicted_labels.append(sentiment)
        else:
            # Pilih model yang sesuai
            if algorithm == 'svm':
                model = self.svm_pipeline
            elif algorithm == 'naive_bayes':
                model = self.nb_pipeline
            else:
                raise ValueError("Algoritma tidak valid. Pilih 'svm', 'naive_bayes', atau 'lexicon'.")
            
            # Lakukan prediksi
            predicted_labels = model.predict(comments)

        # Hitung akurasi
        accuracy = accuracy_score(true_labels, predicted_labels)
        return accuracy * 100

# Contoh penggunaan (opsional, untuk pengujian)
if __name__ == '__main__':
    analyzer = SentimentAnalyzer()
    
    # Menghitung dan menampilkan akurasi
    print("--- Akurasi Model ---")
    svm_accuracy = analyzer.calculate_accuracy(algorithm='svm')
    nb_accuracy = analyzer.calculate_accuracy(algorithm='naive_bayes')
    lexicon_accuracy = analyzer.calculate_accuracy(algorithm='lexicon')
    print(f"Akurasi SVM: {svm_accuracy:.2f}%")
    print(f"Akurasi Naive Bayes: {nb_accuracy:.2f}%")
    print(f"Akurasi Lexicon (VADER): {lexicon_accuracy:.2f}%")

    print("\n" + "="*30 + "\n")

    new_comments = [
        "Video ini luar biasa dan sangat membantu!",
        "Saya tidak mengerti apa-apa, penjelasannya buruk.",
        "Cukup ok, tidak ada yang istimewa."
    ]
    
    # Analisis dengan SVM
    print("--- Hasil Analisis SVM ---")
    svm_results = analyzer.analyze(new_comments, algorithm='svm')
    print(pd.DataFrame(svm_results))
    
    print("\n" + "="*30 + "\n")
    
    # Analisis dengan Naive Bayes
    print("--- Hasil Analisis Naive Bayes ---")
    nb_results = analyzer.analyze(new_comments, algorithm='naive_bayes')
    print(pd.DataFrame(nb_results))

    print("\n" + "="*30 + "\n")

    # Analisis dengan Lexicon (VADER)
    print("--- Hasil Analisis Lexicon-Based (VADER) ---")
    lexicon_results = analyzer.analyze(new_comments, algorithm='lexicon')
    print(pd.DataFrame(lexicon_results))