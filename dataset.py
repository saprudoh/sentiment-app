# dataset.py

def get_test_data():
    """
    Mengembalikan dataset pengujian sederhana dengan sentimen yang sudah diberi label.
    """
    data = [
        # Data Positif
        ("Saya sangat senang dengan produk ini, kualitasnya luar biasa!", "positive"),
        ("Pelayanannya sangat memuaskan, cepat dan ramah.", "positive"),
        ("Film ini benar-benar menginspirasi, saya merekomendasikannya!", "positive"),
        ("Secara keseluruhan, pengalaman yang sangat positif.", "positive"),

        # Data Negatif
        ("Saya sangat kecewa dengan layanan yang diberikan.", "negative"),
        ("Produknya tidak sesuai dengan deskripsi, sangat buruk.", "negative"),
        ("Pengirimannya sangat lambat, saya menunggu lebih dari seminggu.", "negative"),
        ("Jangan buang-buang uang Anda untuk ini, tidak ada gunanya.", "negative"),

        # Data Netral
        ("Ini adalah sebuah pulpen.", "neutral"),
        ("Saya akan pergi ke pasar besok.", "neutral"),
        ("Rapat akan dimulai pukul 10 pagi.", "neutral"),
        ("Sistemnya sekarang sedang offline untuk pemeliharaan.", "neutral")
    ]
    return data
