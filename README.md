<div align="center">

# ✈️ BlackScope: Havacılıkta Yapay Zeka Sistemi
### TEKNOFEST 2026 — Havacılıkta Yapay Zeka Yarışması
#### **Takım:** Darülmu-AI-limin | **Takım ID:** #754906 | **Başvuru ID:** #4983561

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange.svg)](https://pytorch.org/)
[![Ultralytics YOLO](https://img.shields.io/badge/YOLO-v11%20%7C%20v8-00FFFF.svg)](https://github.com/ultralytics/ultralytics)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.8%2B-green.svg)](https://opencv.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![TEKNOFEST](https://img.shields.io/badge/TEKNOFEST-2026-red.svg)](https://www.teknofest.org/)

**[🇹🇷 Türkçe](README.md) | [🇬🇧 English](README_EN.md)**

<p align="center">
  <b>TEKNOFEST 2026 Havacılıkta Yapay Zeka Yarışması için geliştirilmiş, İnsansız Hava Araçları (İHA) kameralarından gelen video akışını tamamen çevrimdışı (offline) ve gerçek zamanlı işleyen çoklu görevli (multitask) bir yapay zeka ve bilgisayarlı görü sistemi.</b>
</p>

</div>

---

## 📌 İçindekiler
- [Proje Genel Bakışı](#-proje-genel-bakışı)
- [Takım Şeması](#-takım-şeması)
- [Temel Yetenekler ve Görev Çözümleri](#-temel-yetenekler-ve-görev-çözümleri)
- [Sistem Mimarisi](#-🏗️-Sistem-Mimarisi)
- [Dizin Yapısı](#-dizin-yapısı)
- [Kurulum](#-🛠️-Kurulum)
- [Kullanım](#-kullanım)
- [Lisans](#-Lisans)

---

## 📌 Proje Genel Bakışı

Yarışma senaryosunda İHA'dan alınan 7.5 FPS video akışı ve eşzamanlı telemetri verileri kullanılarak yarışma sunucusuna saniyede en az 1 kare işleme hızıyla yerel ağ (Ethernet) üzerinden standart JSON formatında yanıt üretilmektedir. 

BlackScope, üç bağımsız yarışma görevini tek bir geçişli (single-pass) entegre boru hattında birleştirir:
1. **Nesne Tespiti, Hareket Analizi ve İniş Alanı Uygunluğu**
2. **Görsel Tabanlı Pozisyon Kestirimi (Görsel Odometri)**
3. **Tanımsız / Referans Nesne Eşleme**

---

## 👥 Takım Şeması

| Rol | Sorumluluk Alanı |
|:---|:---|
| **Danışman (Akademik Koordinatör)** | Yazılım ve yapay zeka alanlarında akademik destek sağlanması. |
| **Takım Kaptanı** | Sistem entegrasyonu, sunucu iletişimi, proje yönetimi ile veri seti derleme, etiketleme araçları, doğrulama test senaryoları. |
| **1. Üye** | YOLOv8/v11 model eğitimi, veri etiketleme, hareket analizi. |
| **2. Üye** | Optik akış, pozisyon kestirim algoritması. |
| **3. Üye** | Öznitelik çıkarma, şablon eşleme, adaptif tanıma. |

---

## 🚀 Temel Yetenekler ve Görev Çözümleri

### 1. Görev 1: Nesne Tespiti, Hareket Analizi ve İniş Uygunluğu
* **YOLOv11 Tabanlı Tespit:** COCO ve VisDrone2019 veri setleri ile öneğitilmiş ve özel İHA görüntüleriyle ince ayar yapılmış model üzerinden 4 sınıf tespit edilir:
  * `0`: Taşıt
  * `1`: İnsan
  * `2`: UAP (Uçan Araba Park Alanı)
  * `3`: UAİ (Uçan Ambulans İniş Alanı)
* **Kamera Ego-Hareketi Telafisi:** İHA'nın uçuşu nedeniyle durağan taşıtların hareketli gibi algılanmasını (sahte pozitif) önlemek için ardışık kareler arasında ORB ve RANSAC ile homografi matrisi ($H$) kestirilir. Kamera hareketi dengelendikten sonra taşıtların yerdeki gerçek hareket durumu (`is_moving`: 1 / 0) belirlenir.
* **İniş Alanı Uygunluğu:** Tespit edilen UAP/UAİ bölgeleri ile taşıtlar arasında IoU (Intersection over Union) analizi gerçekleştirilir. Bölge üzerinde engel varsa iniş durumu `0` (uygun değil), alan boşsa `1` (uygun) olarak raporlanır.

### 2. Görev 2: Görsel Odometri ile Pozisyon Kestirimi
* **GPS Kesintisi Yönetimi:** Telemetride `gps_health_status = 0` olduğunda görsel odometri modülü devreye girer.
* **Lucas-Kanade Piramidal Optik Akış:** Shi-Tomasi algoritmasıyla takip edilecek anahtar noktalar belirlenir, ardışık kareler arasında optik akış izlenir.
* **Poz Kestirimi ve Metrik Ölçeklendirme:** Essential Matrix ($E$) ve RANSAC filtrelemesiyle kameranın rotasyon matrisi ($R$) ve yer değiştirmesi kestirilir. Kamera iç parametre matrisi ($K$) ve irtifa ($Z$) bilgisi ile göreli piksel hareketleri metrik değerlere ($\Delta X, \Delta Y, \Delta Z$) dönüştürülür.
* **Drift Sıfırlama:** GPS sinyali tekrar sağlıklı hale geldiğinde (`health_status = 1`), sistem mutlak konumu sunucu verisiyle senkronize eder.

### 3. Görev 3: Tanımsız / Referans Nesne Eşleme
* **Adaptif ve Hiyerarşik Eşleme:** Yarışma oturumu başında sağlanan referans görsellerin uçuş sırasında tespiti için hibrit bir strateji kullanılır:
  * **Birincil (ORB):** Hızlı ve hesaplama maliyeti düşük öznitelik çıkarımı.
  * **Yedek (SIFT):** Farklı bakış açıları, ölçek değişimleri veya RGB-termal modalite farkları durumunda devreye girer.
* **Homografi ve Bounding Box Doğrulama:** Brute-Force eşleştirme ve Lowe's Ratio Test sonrasında RANSAC homografi uygulanarak nesnenin görüntü koordinatlarındaki kesin sınır kutusu (bounding box) çıkarılır.

---

## 🏗️ Sistem Mimarisi

```text
       ┌────────────────────────────────────────────────────────┐
       │             İHA Video Akışı & Telemetri               │
       └───────────────────────────┬────────────────────────────┘
                                   │
                                   ▼
       ┌────────────────────────────────────────────────────────┐
       │        Ön İşleme (CLAHE, Histogram Eşitleme)           │
       └───────────────────────────┬────────────────────────────┘
                                   │
         ┌─────────────────────────┼────────────────────────┐
         │                         │                        │
         ▼                         ▼                        ▼
┌───────────────────┐    ┌───────────────────┐    ┌───────────────────┐
│     Görev 1       │    │     Görev 2       │    │     Görev 3       │
│ YOLOv11m Tespiti  │    │  Görsel Odometri  │    │  Adaptif Eşleme   │
│   (Taşıt, İnsan,  │    │   (GPS=0 Durumu)  │    │    (ORB / SIFT)   │
│      UAP, UAİ)    │    │   Lucas-Kanade    │    │                   │
│         │         │    │  Essential Matrix │    │ Brute-Force +     │
│         ▼         │    │  Metrik ΔX,ΔY,ΔZ  │    │ Homografi         │
│  Ego-Hareket &    │    └─────────┬─────────┘    └─────────┬─────────┘
│  İniş Uygunluğu   │              │                        │
└────────┬──────────┘              │                        │
         │                         │                        │
         └─────────────────────────┼────────────────────────┘
                                   │
                                   ▼
       ┌────────────────────────────────────────────────────────┐
       │     Entegre JSON Çıktısı (TEKNOFEST Şartnamesi)        │
       └───────────────────────────┬────────────────────────────┘
                                   │
                                   ▼
       ┌────────────────────────────────────────────────────────┐
       │       Yerel Yarışma Sunucusu (HTTP POST /results)       │
       └────────────────────────────────────────────────────────┘
```

---

## 📁 Dizin Yapısı

```text
BlackScope/
├── configs/
│   ├── server_config.json      # Sunucu IP, port ve kamera iç parametreleri
│   └── visdrone.yaml           # Veri seti sınıf tanımları
├── models/
│   └── yolo11m_visdrone.pt     # Eğitilmiş nesne tespit model ağırlıkları
├── results/                    # Test grafikleri ve benchmark çıktıları
│   ├── detection_plot.png
│   ├── matching_plot.png
│   └── odometry_plot.png
├── scripts/
│   ├── run_pipeline.py         # Yarışma modunu veya video testini başlatan ana betik
│   ├── benchmark.py            # Modül bazlı performans ve hız testleri
│   ├── augment_*.py            # Termal, gece ve yağmur simülasyonu betikleri
│   └── train_model.py          # YOLO modeli ince ayar eğitim betiği
├── src/
│   ├── detection/
│   │   ├── ego_motion.py       # Homografi tabanlı kamera ego-hareket telafisi
│   │   ├── landing_analysis.py # UAP / UAİ alanları için engel ve iniş analizi
│   │   ├── preprocessing.py    # CLAHE ve kontrast dengeleme adımları
│   │   └── train_yolo.py       # Model yükleme ve çıkarım (inference)
│   ├── integration/
│   │   ├── pipeline.py         # 3 görevi tek geçişte koşturan ana boru hattı
│   │   └── server_comm.py      # Sunucu iletişim protokolü (GET /frame, POST /results)
│   ├── matching/
│   │   ├── adaptive_match.py   # Hibrit ORB/SIFT eşleme yöneticisi
│   │   ├── orb_matcher.py      # ORB öznitelik çıkarımı ve eşleme
│   │   └── sift_matcher.py     # SIFT öznitelik çıkarımı ve eşleme
│   ├── odometry/
│   │   ├── optical_flow.py     # Lucas-Kanade piramidal optik akış
│   │   └── position_est.py     # Göreli poz ve metrik konum kestirimi
│   └── utils/
│       ├── metrics.py          # IoU hesaplama ve değerlendirme metrikleri
│       └── visualization.py    # Bounding box ve telemetri çizim araçları
├── tests/                      # Otomatik birim testleri
│   ├── test_detection.py
│   ├── test_matching.py
│   ├── test_odometry.py
│   └── test_server.py
├── LICENSE                     # MIT Lisansı
├── requirements.txt            # Python kütüphane bağımlılıkları
└── README.md                   # Proje dökümantasyonu
```

---

## 🛠️ Kurulum

### 1. Depoyu Klonlayın
```bash
git clone https://github.com/kullanici-adi/BlackScope.git
cd BlackScope
```

### 2. Sanal Ortam Oluşturun ve Aktive Edin
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux / macOS
python3 -m venv venv
source venv/bin/activate
```

### 3. Bağımlılıkları Yükleyin
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

> **Not:** Sistem yarışma ortamında tamamen çevrimdışı (offline) çalışmak üzere tasarlanmıştır. Yarışma alanına gitmeden önce tüm pip bağımlılıklarının ve model ağırlıklarının yerel ortamda kurulu olduğundan emin olunuz.

---

## 🚦 Kullanım

### 1. Yarışma Modu (Canlı Sunucu Bağlantısı)
Ethernet üzerinden yarışma sunucusuna bağlanarak görüntüleri çeker, işler ve sonuçları anlık iletir:
```bash
python scripts/run_pipeline.py --mode competition --server http://192.168.1.100:8080
```

### 2. Çevrimdışı Video Test Modu
Kayıtlı bir İHA uçuş videosu üzerinde boru hattını test etmek ve sonuçları JSON olarak kaydetmek için:
```bash
python scripts/run_pipeline.py --mode video --video test_flight.mp4 --output results/
```

### 3. Otomatik Birim Testleri
Tüm sistem bileşenlerinin doğruluğunu test etmek için:
```bash
python -m unittest discover tests/
```

### 4. Başarım ve Hız Kıyaslaması (Benchmark)
Modüllerin FPS ve çıkarım sürelerini ölçmek için:
```bash
python scripts/benchmark.py
```

---

## 📡 Sunucu İletişim Protokolü ve JSON Şeması

Yarışma şartnamesi uyarınca sistem, sunucudan telemetrili görüntü verisini `GET /frame` ile çeker ve işleme sonucunda `POST /results` uç noktasına şu JSON formatında veri gönderir:

```json
{
  "detected_objects": [
    {
      "cls": "0",
      "landing_status": "-1",
      "motion_status": "1",
      "top_left_x": 120,
      "top_left_y": 80,
      "bottom_right_x": 260,
      "bottom_right_y": 190
    },
    {
      "cls": "2",
      "landing_status": "1",
      "motion_status": "-1",
      "top_left_x": 300,
      "top_left_y": 150,
      "bottom_right_x": 480,
      "bottom_right_y": 330
    }
  ],
  "detected_translations": [
    {
      "translation_x": 1.4523,
      "translation_y": -0.8921,
      "translation_z": 12.4012
    }
  ],
  "detected_undefined_objects": [
    {
      "object_id": "target_1",
      "top_left_x": 210,
      "top_left_y": 140,
      "bottom_right_x": 310,
      "bottom_right_y": 240
    }
  ]
}
```

* `cls`: `0` (Taşıt), `1` (İnsan), `2` (UAP), `3` (UAİ)
* `landing_status`: UAP/UAİ için `1` (uygun) veya `0` (dolu), diğer sınıflar için `-1`
* `motion_status`: Taşıt için `1` (hareketli) veya `0` (durağan), diğer sınıflar için `-1`
* `detected_translations`: Yalnızca `gps_health_status = 0` durumunda görsel odometri ile üretilir.

---

## 📄 Lisans

Bu proje [MIT Lisansı](LICENSE) altında lisanslanmıştır. Detaylar için `LICENSE` dosyasını inceleyebilirsiniz.
