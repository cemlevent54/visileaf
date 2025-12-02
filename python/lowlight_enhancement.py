import numpy as np
import cv2

from scipy.spatial import distance
from scipy.ndimage import convolve
from scipy.sparse import diags, csr_matrix
from scipy.sparse.linalg import spsolve


def get_sparse_neighbor(p: int, n: int, m: int):
  """
  Komşuluk bilgilerini döndürür.

  Orijinal LIME/DUAL implementasyonundaki `get_sparse_neighbor` fonksiyonuna
  benzer şekilde, her piksel için 4-komşulu (yukarı, aşağı, sol, sağ) komşuları
  ve bunların yön bilgisini (yatay/dikey) üretir.

  Args:
      p: Düzleştirilmiş indeks (0..n*m-1)
      n: Yükseklik
      m: Genişlik

  Returns:
      dict[int, tuple[int, int, bool]]:
          q -> (k, l, x) şeklinde:
            - q: komşu pikselin düzleştirilmiş indeksi
            - k, l: komşu pikselin satır ve sütunu
            - x: True ise yatay (left/right), False ise dikey (up/down)
  """
  i = p // m
  j = p % m
  neighbors = {}

  # Sol komşu (yatay)
  if j - 1 >= 0:
    q = p - 1
    neighbors[q] = (i, j - 1, True)

  # Sağ komşu (yatay)
  if j + 1 < m:
    q = p + 1
    neighbors[q] = (i, j + 1, True)

  # Üst komşu (dikey)
  if i - 1 >= 0:
    q = p - m
    neighbors[q] = (i - 1, j, False)

  # Alt komşu (dikey)
  if i + 1 < n:
    q = p + m
    neighbors[q] = (i + 1, j, False)

  return neighbors


def create_spacial_affinity_kernel(spatial_sigma: float, size: int = 15) -> np.ndarray:
  """
  Uzamsal yakınlık temelli Gaussian ağırlık kerneli oluşturur.
  """
  kernel = np.zeros((size, size))
  center = (size // 2, size // 2)

  for i in range(size):
    for j in range(size):
      d = distance.euclidean((i, j), center)
      kernel[i, j] = np.exp(-0.5 * (d ** 2) / (spatial_sigma ** 2))

  return kernel


def compute_smoothness_weights(L: np.ndarray, x: int, kernel: np.ndarray, eps: float = 1e-3) -> np.ndarray:
  """
  Aydınlatma haritası için düzgünlük ağırlıklarını hesaplar.

  Args:
      L: Başlangıç aydınlatma haritası
      x: Yön (1: yatay, 0: dikey)
      kernel: Uzamsal affinity matrisi
      eps: Sayısal kararlılık için küçük sabit
  """
  Lp = cv2.Sobel(L, cv2.CV_64F, int(x == 1), int(x == 0), ksize=1)

  T = convolve(np.ones_like(L), kernel, mode="constant")
  T = T / (np.abs(convolve(Lp, kernel, mode="constant")) + eps)

  return T / (np.abs(Lp) + eps)


def fuse_multi_exposure_images(
  im: np.ndarray,
  under_ex: np.ndarray,
  over_ex: np.ndarray,
  bc: float = 1.0,
  bs: float = 1.0,
  be: float = 1.0,
) -> np.ndarray:
  """
  DUAL makalesindeki exposure fusion yöntemini uygular.
  """
  merge_mertens = cv2.createMergeMertens(bc, bs, be)

  images = [np.clip(x * 255, 0, 255).astype("uint8") for x in [im, under_ex, over_ex]]
  fused_images = merge_mertens.process(images)

  return fused_images


def refine_illumination_map_linear(
  L: np.ndarray,
  gamma: float,
  lambda_: float,
  kernel: np.ndarray,
  eps: float = 1e-3,
) -> np.ndarray:
  """
  LIME/DUAL'de tanımlanan optimizasyon problemini çözerek aydınlatma
  haritasını rafine eder (hızlandırılmış lineer çözücü).
  """
  # Düzgünlük ağırlıkları
  wx = compute_smoothness_weights(L, x=1, kernel=kernel, eps=eps)
  wy = compute_smoothness_weights(L, x=0, kernel=kernel, eps=eps)

  n, m = L.shape
  L_1d = L.copy().flatten()

  # Beş-noktalı, uzamsal olarak inhomogeneous Laplace matrisi
  row, column, data = [], [], []

  for p in range(n * m):
    diag = 0.0
    for q, (k, l, is_horizontal) in get_sparse_neighbor(p, n, m).items():
      weight = wx[k, l] if is_horizontal else wy[k, l]
      row.append(p)
      column.append(q)
      data.append(-weight)
      diag += weight

    row.append(p)
    column.append(p)
    data.append(diag)

  F = csr_matrix((data, (row, column)), shape=(n * m, n * m))

  # Lineer sistemi çöz
  Id = diags([np.ones(n * m)], [0])
  A = Id + lambda_ * F

  L_refined = spsolve(csr_matrix(A), L_1d).reshape((n, m))

  # Gamma düzeltmesi
  L_refined = np.clip(L_refined, eps, 1.0) ** gamma

  return L_refined


def correct_underexposure(
  im: np.ndarray,
  gamma: float,
  lambda_: float,
  kernel: np.ndarray,
  eps: float = 1e-3,
) -> np.ndarray:
  """
  LIME/DUAL'deki retinex-tabanlı algoritma ile düşük pozlanmış bölgeleri düzeltir.
  """
  # İlk aydınlatma haritası tahmini
  L = np.max(im, axis=-1)

  # Aydınlatma haritasını rafine et
  L_refined = refine_illumination_map_linear(L, gamma, lambda_, kernel, eps)

  # Görüntüyü düzelt
  L_refined_3d = np.repeat(L_refined[..., None], 3, axis=-1)
  im_corrected = im / L_refined_3d

  return im_corrected


def enhance_image_exposure(
  im: np.ndarray,
  gamma: float,
  lambda_: float,
  dual: bool = True,
  sigma: int = 3,
  bc: float = 1.0,
  bs: float = 1.0,
  be: float = 1.0,
  eps: float = 1e-3,
) -> np.ndarray:
  """
  Girdi görüntüsünün pozlamasını LIME veya DUAL yöntemleriyle iyileştirir.
  """
  # Uzamsal affinity kerneli
  kernel = create_spacial_affinity_kernel(sigma)

  # Normalize et
  im_normalized = im.astype(float) / 255.0

  # Düşük pozlanmış bölgeleri düzelt
  under_corrected = correct_underexposure(im_normalized, gamma, lambda_, kernel, eps)

  if dual:
    # DUAL için: aşırı pozlanmış bölgeleri de düzelt ve birleştir
    inv_im_normalized = 1.0 - im_normalized
    over_corrected = 1.0 - correct_underexposure(inv_im_normalized, gamma, lambda_, kernel, eps)

    im_corrected = fuse_multi_exposure_images(im_normalized, under_corrected, over_corrected, bc, bs, be)
  else:
    # LIME için yalnızca under_corrected kullanılır
    im_corrected = under_corrected

  # 8-bit aralığa geri dön
  return np.clip(im_corrected * 255.0, 0, 255).astype("uint8")


