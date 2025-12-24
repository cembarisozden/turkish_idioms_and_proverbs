# Kurulum Talimatları

## Bağımlılıkları Yükleme

Python 3.13.1 yüklü ancak pip modülü bulunamıyor. Aşağıdaki adımları izleyin:

### Seçenek 1: pip'i yükleyin (önerilen)

1. PowerShell'de şu komutu çalıştırın:
```powershell
python -m ensurepip --upgrade
```

2. Ardından paketleri yükleyin:
```powershell
python -m pip install torch transformers scikit-learn pandas numpy openpyxl
```

### Seçenek 2: get-pip.py kullanın

1. https://bootstrap.pypa.io/get-pip.py adresinden `get-pip.py` dosyasını indirin
2. PowerShell'de:
```powershell
python get-pip.py
python -m pip install torch transformers scikit-learn pandas numpy openpyxl
```

### Seçenek 3: Anaconda/Miniconda kullanın

Eğer Anaconda yüklüyse:
```powershell
conda install pandas numpy scikit-learn
pip install torch transformers openpyxl
```

## Kurulum Sonrası

Kurulum tamamlandıktan sonra:
```powershell
python scripts/run_prepare_data.py
```

