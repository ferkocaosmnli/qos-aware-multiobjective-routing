# 📡 Akıllı QoS Yönlendirme ve Algoritma Karşılaştırma Paneli

## 📌 Proje Tanımı

Bu proje, **Quality of Service (QoS) farkındalıklı ağ yönlendirme** problemini ele alan bir karar destek ve analiz sistemidir.  
Amaç, bir ağ topolojisi üzerinde **kaynak–hedef** düğümleri arasında, **gecikme, güvenilirlik ve maliyet** gibi çoklu QoS kriterlerini dikkate alarak **en uygun yönlendirme yolunu** hesaplamaktır.

Projede farklı optimizasyon ve pekiştirmeli öğrenme algoritmaları aynı problem üzerinde çalıştırılarak **karşılaştırmalı analiz** yapılabilmektedir.

---

## 🎯 Temel Özellikler

- Çok kriterli QoS tabanlı yönlendirme
- Birden fazla algoritmanın karşılaştırılması
- Tekrarlanabilir (reproducible) sonuçlar için seed kullanımı
- İnteraktif ağ görselleştirmesi
- Web tabanlı kullanıcı arayüzü (Streamlit)

---

## ⚙️ Kullanılan Algoritmalar

Projede aşağıdaki algoritmalar uygulanmıştır:

- **Genetic Algorithm (GA)**  
  Evrimsel optimizasyon yaklaşımı ile rota seçimi

- **Q-Learning**  
  Pekiştirmeli öğrenme temelli yol bulma

- **SARSA**  
  On-policy pekiştirmeli öğrenme yöntemi

- **Simulated Annealing (SA)**  
  Yerel minimumlardan kaçınmayı hedefleyen stokastik optimizasyon

Tüm algoritmalar **aynı QoS maliyet fonksiyonunu** kullanır; bu sayede adil bir karşılaştırma yapılır.

---

## 📊 QoS Kriterleri

Her rota için aşağıdaki metrikler hesaplanır:

- ⏱ **Gecikme (Delay)**
- 🛡 **Güvenilirlik (Reliability)**
- 💰 **Kaynak/Maliyet (Resource Cost)**

Kullanıcı, bu kriterlere **ağırlık** vererek çok kriterli bir skor oluşturur:

```python
weights = {
    "delay": 0.5,
    "reliability": 0.3,
    "resource": 0.2
}
```

---

## 🧠 Seed (Tekrarlanabilirlik)

Deneylerin her çalıştırmada aynı sonucu vermesi için **rastgelelik kontrol altına alınmıştır**.

Kod içerisinde kullanılan seed değeri:

```python
import random
import numpy as np

random.seed(42)
np.random.seed(42)
```

Bu sayede:
- Algoritmalar karşılaştırılabilir
- Sonuçlar tekrar üretilebilir
- Akademik deney disiplini sağlanır

---

## 📂 Proje Klasör Yapısı

```
network_sim_project_v5/
│
├── src/
│   ├── algorithms/        # GA, Q-Learning, SARSA, SA
│   ├── experiments/       # Senaryo ve deney akışı
│   ├── network/           # Graph ve CSV yükleme
│   ├── web_gui.py         # Streamlit arayüzü
│
├── graph_csv/
│   ├── NodeData.csv
│   ├── EdgeData.csv
│   └── DemandData.csv
│
├── requirements.txt
├── README.md
└── venv/ (opsiyonel)
```

---

## ▶️ Çalıştırma Adımları

### 1️⃣ Sanal Ortam Oluşturma (İlk kez)

```bash
python3 -m venv venv
source venv/bin/activate
```

### 2️⃣ Gerekli Paketlerin Kurulumu

```bash
pip install -r requirements.txt
```

> Eğer `requirements.txt` yoksa:
```bash
pip install streamlit pandas networkx plotly numpy matplotlib
```

### 3️⃣ Uygulamanın Başlatılması

```bash
streamlit run src/web_gui.py
```

Tarayıcıda otomatik açılmazsa:
```
http://localhost:8501
```

---

## 🖥 Kullanıcı Arayüzü

Uygulama üç ana analiz bölümünden oluşur:

### 🔍 Analiz
- Tek algoritma
- Tek senaryo
- Hesaplanan rota ve QoS metrikleri

### 📊 Pareto Analizi
- Aynı algoritma
- Farklı ağırlık kombinasyonları
- Çok kriterli karar analizi

### ⚔️ Kıyaslama (Performans Arenası)
- İki algoritmanın karşılaştırılması
- Süre, skor ve rota analizi

---

## 🎬 Demo ve Sunum

- Proje, Streamlit tabanlı arayüz ile **canlı olarak çalıştırılabilir**
- Kısa bir demo videosu ile işleyiş gösterilebilir
- Sınıf ortamında canlı sunum ve soru–cevap için uygundur

---

## 📌 Sonuç

Bu proje, QoS farkındalıklı yönlendirme problemini:
- Algoritmik,
- Deneysel,
- Görsel

olarak ele alan bütünleşik bir sistem sunmaktadır.  
Gerçekçi ağ senaryolarında farklı algoritmaların performanslarını karşılaştırmak için kullanılabilir.

Bu proje 
Rana AKYÜZ
Ferhat KOCAOSMANLI
Sıla TOKER
Hasan TOKPINAR
Zekiye ILMAN
Buse GÜVEZ
Yaren Deniz TEZCAN
Dheya ALESHAWI tarafından oluşturulmuş, geliştirilmiştir.

<img width="470" height="286" alt="image" src="https://github.com/user-attachments/assets/a99dd50d-ff36-47bd-9305-4ec142148c4f" />
