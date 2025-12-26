# Hızlandırma Optimizasyonları

Eğitimi hızlandırmak için yapılan değişiklikler:

## Yapılan Optimizasyonlar

1. **MAX_LENGTH: 128 → 64**
   - Daha kısa sequence'ler = daha hızlı işleme
   - ~2x hızlanma

2. **BATCH_SIZE: 16 → 32**
   - Daha büyük batch = daha az iterasyon
   - ~2x hızlanma
   - Not: CPU bellek sınırına dikkat edin

3. **NUM_EPOCHS: 3 → 2**
   - Daha az epoch = daha hızlı tamamlanma
   - ~33% hızlanma

4. **Training örnekleri: 5000 → 3000**
   - Daha az veri = daha hızlı eğitim
   - ~40% hızlanma

## Toplam Etki

Yaklaşık **4-5x daha hızlı** eğitim bekleniyor.

## CPU'da Daha Hızlı İçin Ek Öneriler

1. **Batch size'ı daha da artırın** (bellek izin veriyorsa):
   ```python
   BATCH_SIZE = 64  # config.py'de
   ```

2. **Sadece 1 epoch ile test edin**:
   ```python
   NUM_EPOCHS = 1  # config.py'de
   ```

3. **Daha az örnek kullanın**:
   ```python
   NUM_POSITIVE_EXAMPLES = 2000
   NUM_NEGATIVE_EXAMPLES = 2000
   ```

## GPU Kullanımı

GPU varsa çok daha hızlı olur. CUDA kuruluysa otomatik kullanılır.


