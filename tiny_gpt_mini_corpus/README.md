# TinyGPT Mini Corpus - EduRisk

Modul ini merupakan tambahan pada repository EduRisk_EBM_Project untuk memenuhi tugas Proyek Data Mining: Membangun TinyGPT dengan Mini Corpus.

## Deskripsi Corpus

Corpus yang digunakan bernama corpus_edurisk.txt. Domain corpus adalah pengelolaan risiko akademik mahasiswa. Isi corpus membahas indikator risiko akademik, performa mahasiswa, peran dosen wali, sistem peringatan dini, data mining pendidikan, evaluasi akademik, dan penggunaan teknologi informasi dalam pemantauan studi mahasiswa.

Jumlah kata corpus: 2.151 kata.

## Struktur Folder

tiny_gpt_mini_corpus/
- corpus/corpus_edurisk.txt
- src/transformer_blocks.py
- src/train_word_tokenizer.py
- src/train_bpe_tokenizer.py
- output/hasil_word_tokenizer.txt
- output/hasil_bpe_tokenizer.txt
- output/edurisk_bpe.model
- output/edurisk_bpe.vocab

## Pendekatan Tokenisasi

Eksperimen dilakukan menggunakan dua pendekatan tokenisasi.

1. Word-Level Tokenization
Teks dipecah berdasarkan kata utuh menggunakan regular expression.

2. BPE Tokenization / SentencePiece
Teks dipecah menjadi subword menggunakan SentencePiece dengan vocabulary size 200.

## Cara Menjalankan

Install library:
pip install torch sentencepiece

Jalankan Word-Level:
python tiny_gpt_mini_corpus/src/train_word_tokenizer.py

Jalankan BPE:
python tiny_gpt_mini_corpus/src/train_bpe_tokenizer.py

## Ringkasan Hasil

Word-Level Tokenization menghasilkan teks yang lebih mudah dibaca karena setiap token berupa kata utuh. Namun, hasil loss menunjukkan indikasi overfitting karena train loss turun tajam sedangkan validation loss meningkat.

BPE Tokenization lebih efisien karena vocabulary dibatasi menjadi 200 token. Hasil loss lebih stabil karena train loss dan validation loss sama-sama menurun. Namun, hasil generate masih memuat beberapa potongan kata yang kurang sempurna karena model kecil dilatih pada corpus terbatas.

## Kesimpulan

Pada corpus kecil bertema risiko akademik mahasiswa, Word-Level Tokenization unggul dalam keterbacaan teks, sedangkan BPE Tokenization unggul dalam efisiensi vocabulary dan stabilitas loss.
