# Julia Machine Learning Project

Bu proje, Julia programlama dili kullanarak temel makine öğrenmesi algoritmalarını göstermek için tasarlanmış basit bir örnektir. Proje, mock veri üretimi, veri ön işleme, karar ağacı tabanlı sınıflandırma ve regresyon modellerini içerir.

## Gereksinimler

Bu projeyi çalıştırmak için aşağıdaki Julia paketlerinin yüklü olması gerekmektedir:

```julia
using Pkg
Pkg.add("Random")
Pkg.add("DataFrames")
Pkg.add("MLDataUtils")
Pkg.add("DecisionTree")
Pkg.add("Statistics")
Pkg.add("Plots")
```

## Proje Yapısı

### Ana Bileşenler

1. **Mock Veri Üretimi** (`generate_mock_data`)
2. **Veri Ön İşleme** (`preprocess`)
3. **Sınıflandırma Modeli** (`train_classification`)
4. **Regresyon Modeli** (`train_regression`)

## Fonksiyon Açıklamaları

### `generate_mock_data(n::Int=500)`
- **Amaç**: Yapay veri seti oluşturur
- **Parametreler**: `n` - oluşturulacak örneklerin sayısı (varsayılan: 500)
- **Çıktı**: 3 özellik (x1, x2, x3) ve 2 hedef değişken (y_class, y_reg) içeren DataFrame
- **Özellikler**:
  - x1, x2, x3: Normal dağılımdan örneklenmiş özellikler
  - y_class: Binary sınıflandırma hedefi (0 veya 1)
  - y_reg: Sürekli regresyon hedefi

### `preprocess(df::DataFrame)`
- **Amaç**: Veri setini eğitim ve test setlerine ayırır
- **Bölünme Oranı**: %70 eğitim, %30 test
- **Çıktı**: (train, test) tuple'ı

### `train_classification(train, test)`
- **Amaç**: Karar ağacı sınıflandırıcısı eğitir ve test eder
- **Model**: DecisionTreeClassifier (max_depth=5)
- **Metrik**: Doğruluk (Accuracy)
- **Çıktı**: Eğitilmiş sınıflandırma modeli

### `train_regression(train, test)`
- **Amaç**: Karar ağacı regresyonu eğitir ve test eder
- **Model**: DecisionTreeRegressor (max_depth=5)
- **Metrik**: RMSE (Root Mean Square Error)
- **Çıktı**: Eğitilmiş regresyon modeli

## Kullanım

Projeyi çalıştırmak için:

1. Gerekli paketlerin yüklü olduğundan emin olun
2. Kodu Julia REPL'de veya bir .jl dosyası olarak çalıştırın

```julia
# Veri üretimi
df = generate_mock_data(1000)

# Veri ön işleme
train, test = preprocess(df)

# Model eğitimi ve değerlendirme
clf_model = train_classification(train, test)
reg_model = train_regression(train, test)
```

## Beklenen Çıktılar

Program çalıştırıldığında aşağıdaki gibi sonuçlar üretir:

```
Classification Accuracy: 0.85 (örnek değer)
Regression RMSE: 1.23 (örnek değer)
```

## Geliştirilme Önerileri

- Farklı makine öğrenmesi algoritmalarının eklenmesi
- Cross-validation implementasyonu
- Hiperparametre optimizasyonu
- Veri görselleştirme özelliklerinin genişletilmesi
- Model performans karşılaştırmaları

## Lisans

Bu proje eğitim amaçlı olarak geliştirilmiştir ve serbestçe kullanılabilir.

## Katkıda Bulunma

Projeyi geliştirmek için pull request'ler ve issue'lar memnuniyetle karşılanır.
