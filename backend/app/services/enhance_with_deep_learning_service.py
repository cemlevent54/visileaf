"""
Deep learning tabanlı hazır modeller için servis katmanı.

Not: Şimdilik sadece EnlightenGAN destekleniyor. İleride model bazlı
implementasyonlar burada genişletilecek.
"""
import sys
import os
import shutil
import tempfile
import io
from pathlib import Path
from typing import Callable, Dict, Optional
import logging
from PIL import Image
import cv2
import numpy as np
import torch
import yaml

logger = logging.getLogger(__name__)


class EnhanceWithDeepLearningService:
    """
    Deep learning modelleriyle görüntü geliştirme servisidir.
    `model_name` değerine göre ilgili test/inference metodu çağrılır.
    """

    def __init__(self):
        self._dispatch: Dict[str, Callable[[bytes, str], bytes]] = {
            "enlightengan": self._run_enlightengan,
            "zero_dce": self._run_zero_dce,
            "llflow": self._run_llflow,
        }

    def enhance_with_model(self, *, image_bytes: bytes, model_name: str, original_filename: str) -> bytes:
        """
        Verilen model adına göre ilgili DL pipeline'ını çalıştırır.
        """
        key = (model_name or "").strip().lower()
        handler = self._dispatch.get(key)
        if handler is None:
            raise ValueError(f"Desteklenmeyen model: {model_name}")

        return handler(image_bytes, original_filename)

    # ----------------------
    # EnlightenGAN helpers
    # ----------------------
    def _get_enlightengan_root(self) -> Path:
        """
        EnlightenGAN kök klasörü (backend/app/helpers/EnlightenGAN).
        """
        return Path(__file__).parent.parent / "helpers" / "EnlightenGAN"

    def _enlightengan_sync_weights(self, root: Path) -> None:
        """
        weight_and_models içindeki hazır ağırlıkları beklenen konumlara kopyalar.
        - Generator: checkpoints/enlightening/200_net_G_A.pth
        - VGG: model/vgg16.weight
        """
        # Ağırlıkların kaynak konumu: backend/app/weight_and_models/EnlightenGAN
        weights_src_dir = Path(__file__).parent.parent / "weight_and_models" / "EnlightenGAN"
        
        gen_src = weights_src_dir / "200_net_G_A.pth"
        gen_dst = root / "checkpoints" / "enlightening" / "200_net_G_A.pth"
        if gen_src.exists():
            gen_dst.parent.mkdir(parents=True, exist_ok=True)
            if not gen_dst.exists():
                shutil.copy(gen_src, gen_dst)
                logger.info("EnlightenGAN ağırlığı kopyalandı: %s", gen_dst)
        else:
            logger.warning("EnlightenGAN generator ağırlığı bulunamadı: %s", gen_src)

        vgg_src = weights_src_dir / "vgg16.weight"
        vgg_dst = root / "model" / "vgg16.weight"
        if vgg_src.exists():
            vgg_dst.parent.mkdir(parents=True, exist_ok=True)
            if not vgg_dst.exists():
                shutil.copy(vgg_src, vgg_dst)
                logger.info("EnlightenGAN VGG ağırlığı kopyalandı: %s", vgg_dst)
        else:
            logger.warning("EnlightenGAN vgg16.weight bulunamadı: %s", vgg_src)

    def _enlightengan_prepare_test_folders(self, root: Path, input_image: Path, keep_existing: bool = False) -> Path:
        """
        EnlightenGAN predict modu testA/testB klasörlerini zorunlu kılıyor.
        testA: girdi görselleri, testB: en az bir görsel (dummy kabul ediliyor).
        """
        test_root = root / "test_dataset"
        testA = test_root / "testA"
        testB = test_root / "testB"
        if test_root.exists() and not keep_existing:
            shutil.rmtree(test_root)
        testA.mkdir(parents=True, exist_ok=True)
        testB.mkdir(parents=True, exist_ok=True)
        target_input = testA / input_image.name
        shutil.copy(input_image, target_input)
        dummy_target = testB / f"dummy_{input_image.name}"
        shutil.copy(input_image, dummy_target)
        return test_root

    def _enlightengan_build_options(self, root: Path):
        """
        predict inline akışı için TestOptions kurar (predict.py ile aynı argümanlar).
        """
        argv = [
            "predict_inline",
            "--dataroot",
            str(root / "test_dataset"),
            "--name",
            "enlightening",
            "--model",
            "single",
            "--which_direction",
            "AtoB",
            "--no_dropout",
            "--dataset_mode",
            "unaligned",
            "--which_model_netG",
            "sid_unet_resize",
            "--skip",
            "1",
            "--use_norm",
            "1",
            "--use_wgan",
            "0",
            "--self_attention",
            "--times_residual",
            "--instance_norm",
            "0",
            "--resize_or_crop",
            "no",
            "--which_epoch",
            "200",
            "--results_dir",
            str(root / "results"),
            "--checkpoints_dir",
            str(root),  # checkpoints_dir root olarak ayarla (checkpoints/ alt klasörü otomatik eklenir)
            "--gpu_ids",
            "-1",  # CPU modu (-1 = CPU, 0+ = GPU ID)
            "--display_id",
            "0",  # Visdom'u devre dışı bırak (0 = görselleştirme yok)
        ]

        orig_argv = sys.argv
        sys.argv = argv
        try:
            from options.test_options import TestOptions

            opt = TestOptions().parse()
        finally:
            sys.argv = orig_argv

        # CPU modunu garanti et (CUDA yoksa veya gpu_ids -1 ise)
        import torch
        if not torch.cuda.is_available() or (hasattr(opt, 'gpu_ids') and opt.gpu_ids == [-1]):
            opt.gpu_ids = []
            opt.device = torch.device('cpu')
        else:
            opt.device = torch.device(f'cuda:{opt.gpu_ids[0]}' if opt.gpu_ids else 'cpu')

        # checkpoints_dir'yi mutlak yol olarak garanti et
        if not os.path.isabs(opt.checkpoints_dir):
            opt.checkpoints_dir = str(root / opt.checkpoints_dir.lstrip('./'))
        else:
            opt.checkpoints_dir = str(root / "checkpoints")

        opt.nThreads = 1
        opt.batchSize = 1
        opt.serial_batches = True
        opt.no_flip = True
        return opt

    def _enlightengan_run_predict_inline(self, root: Path) -> None:
        """
        predict.py akışını inline olarak çalıştırır.
        """
        # repo içindeki modüllerin import edilebilmesi için sys.path'e ekle
        root_str = str(root)
        if root_str not in sys.path:
            sys.path.insert(0, root_str)

        from data.data_loader import CreateDataLoader
        from models.models import create_model
        from util import html
        from util.visualizer import Visualizer

        opt = self._enlightengan_build_options(root)

        data_loader = CreateDataLoader(opt)
        dataset = data_loader.load_data()
        model = create_model(opt)
        visualizer = Visualizer(opt)
        web_dir = os.path.join(opt.results_dir, opt.name, f"{opt.phase}_{opt.which_epoch}")
        webpage = html.HTML(web_dir, f"Experiment = {opt.name}, Phase = {opt.phase}, Epoch = {opt.which_epoch}")

        for _, data in enumerate(dataset):
            model.set_input(data)
            visuals = model.predict()
            img_path = model.get_image_paths()
            visualizer.save_images(webpage, visuals, img_path)

        webpage.save()

    def _run_enlightengan(self, image_bytes: bytes, original_filename: str) -> bytes:
        """
        EnlightenGAN test/predict çalıştırır, çıktı bytes döner.
        """
        root = self._get_enlightengan_root()
        if not root.exists():
            raise ValueError(f"EnlightenGAN kökü bulunamadı: {root}")

        suffix = Path(original_filename).suffix or ".jpg"
        tmp_dir = Path(tempfile.mkdtemp(prefix="enlightengan_", dir=root))
        input_path = tmp_dir / f"input{suffix}"
        input_path.write_bytes(image_bytes)

        self._enlightengan_sync_weights(root)
        self._enlightengan_prepare_test_folders(root, input_path, keep_existing=False)

        try:
            self._enlightengan_run_predict_inline(root)
        except Exception as exc:  # geniş tutuyoruz; predict sırasında farklı exception'lar gelebilir
            logger.exception("EnlightenGAN predict başarısız")
            raise ValueError(f"EnlightenGAN predict başarısız: {exc}") from exc

        results_dir = root / "results"
        if not results_dir.exists():
            raise ValueError(f"EnlightenGAN sonuç klasörü bulunamadı: {results_dir}")

        # EnlightenGAN çıktısı results/enlightening/test_200/images/ altında PNG olarak kaydediliyor
        # Sadece resim dosyalarını (PNG, JPG, JPEG) filtrele
        image_extensions = {'.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG'}
        candidates = [
            p for p in results_dir.glob("**/*")
            if p.is_file() and p.suffix in image_extensions
        ]
        
        if not candidates:
            raise ValueError("EnlightenGAN çıktı resmi bulunamadı.")
        
        # En son değiştirilmiş resim dosyasını al
        output_file = max(candidates, key=lambda p: p.stat().st_mtime)
        
        output_bytes = output_file.read_bytes()
        
        # PNG ise JPEG'e çevir (ImageService JPEG bekliyor)
        if output_file.suffix.lower() in {'.png', '.PNG'}:
            img = Image.open(io.BytesIO(output_bytes))
            # RGB'ye çevir (PNG'de RGBA olabilir)
            if img.mode in ('RGBA', 'LA', 'P'):
                background = Image.new('RGB', img.size, (255, 255, 255))
                if img.mode == 'P':
                    img = img.convert('RGBA')
                background.paste(img, mask=img.split()[-1] if img.mode == 'RGBA' else None)
                img = background
            elif img.mode != 'RGB':
                img = img.convert('RGB')
            
            # JPEG'e çevir
            jpeg_buffer = io.BytesIO()
            img.save(jpeg_buffer, format='JPEG', quality=95)
            output_bytes = jpeg_buffer.getvalue()

        try:
            shutil.rmtree(tmp_dir, ignore_errors=True)
        except Exception:
            pass

        return output_bytes

    # ----------------------
    # Zero-DCE helpers
    # ----------------------
    def _get_zero_dce_root(self) -> Path:
        """
        Zero-DCE kök klasörü (backend/app/helpers/Zero-DCE).
        """
        return Path(__file__).parent.parent / "helpers" / "Zero-DCE"

    def _zero_dce_sync_weights(self, root: Path) -> Path:
        """
        weight_and_models içindeki hazır ağırlıkları beklenen konuma kopyalar.
        - Model: Zero-DCE_code/Epoch99.pth (veya benzeri)
        
        Returns:
            Model ağırlık dosyasının yolu
        """
        weights_src_dir = Path(__file__).parent.parent / "weight_and_models" / "Zero-DCE"
        
        # Model dosyasını bul (.pth uzantılı)
        model_files = list(weights_src_dir.glob("*.pth"))
        if not model_files:
            raise ValueError(f"Zero-DCE model ağırlığı bulunamadı: {weights_src_dir}")
        
        # İlk .pth dosyasını kullan (veya en son olanı)
        model_src = max(model_files, key=lambda p: p.stat().st_mtime)
        
        # Zero-DCE_code klasörüne kopyala
        zero_dce_code_dir = root / "Zero-DCE_code"
        zero_dce_code_dir.mkdir(parents=True, exist_ok=True)
        
        model_dst = zero_dce_code_dir / model_src.name
        if not model_dst.exists():
            shutil.copy(model_src, model_dst)
            logger.info("Zero-DCE ağırlığı kopyalandı: %s", model_dst)
        
        return model_dst

    def _run_zero_dce(self, image_bytes: bytes, original_filename: str) -> bytes:
        """
        Zero-DCE test/inference çalıştırır, çıktı bytes döner.
        """
        root = self._get_zero_dce_root()
        if not root.exists():
            raise ValueError(f"Zero-DCE kökü bulunamadı: {root}")

        # Geçici dosyalar için klasör oluştur
        suffix = Path(original_filename).suffix or ".jpg"
        tmp_dir = Path(tempfile.mkdtemp(prefix="zero_dce_", dir=root))
        input_path = tmp_dir / f"input{suffix}"
        output_path = tmp_dir / "output.jpg"
        
        # Input görseli kaydet
        input_path.write_bytes(image_bytes)

        try:
            # Ağırlıkları senkronize et
            model_path = self._zero_dce_sync_weights(root)

            # Zero-DCE_code klasörünü path'e ekle
            zero_dce_code_dir = root / "Zero-DCE_code"
            zero_dce_code_str = str(zero_dce_code_dir)
            if zero_dce_code_str not in sys.path:
                sys.path.insert(0, zero_dce_code_str)

            # Model modülünü import et
            import model as zero_dce_model
            import torch
            import torchvision
            import numpy as np

            # Device ayarı (CPU modu - GPU yoksa)
            device = torch.device('cpu')
            if torch.cuda.is_available():
                # GPU varsa kullanabiliriz ama şimdilik CPU'da çalıştırıyoruz
                # device = torch.device('cuda')
                pass

            logger.info(f"Zero-DCE Device: {device}")
            logger.info(f"Zero-DCE Loading model from: {model_path}")

            # Modeli yükle
            DCE_net = zero_dce_model.enhance_net_nopool()
            DCE_net.load_state_dict(torch.load(str(model_path), map_location=device))
            DCE_net.to(device)
            DCE_net.eval()

            # Görseli yükle ve ön işle
            logger.info(f"Zero-DCE Loading image: {input_path}")
            data_lowlight = Image.open(input_path)

            # RGB'ye çevir (eğer RGBA veya grayscale ise)
            if data_lowlight.mode != 'RGB':
                data_lowlight = data_lowlight.convert('RGB')

            # Normalize et [0, 1] aralığına
            data_lowlight = np.asarray(data_lowlight) / 255.0

            # Tensor'a çevir ve batch dimension ekle
            data_lowlight = torch.from_numpy(data_lowlight).float()
            data_lowlight = data_lowlight.permute(2, 0, 1)  # HWC -> CHW
            data_lowlight = data_lowlight.unsqueeze(0)  # Batch dimension ekle
            data_lowlight = data_lowlight.to(device)

            # Inference
            logger.info("Zero-DCE Processing image...")
            with torch.no_grad():
                _, enhanced_image, _ = DCE_net(data_lowlight)

            # Sonucu kaydet
            logger.info(f"Zero-DCE Saving result to: {output_path}")
            torchvision.utils.save_image(enhanced_image, output_path)

            # Çıktıyı oku
            output_bytes = output_path.read_bytes()

            # Temizlik
            try:
                shutil.rmtree(tmp_dir, ignore_errors=True)
            except Exception:
                pass

            return output_bytes

        except Exception as exc:
            logger.exception("Zero-DCE predict başarısız")
            # Temizlik
            try:
                shutil.rmtree(tmp_dir, ignore_errors=True)
            except Exception:
                pass
            raise ValueError(f"Zero-DCE predict başarısız: {exc}") from exc

    # ----------------------
    # LLFlow helpers
    # ----------------------
    def _get_llflow_root(self) -> Path:
        """
        LLFlow kök klasörü (backend/app/helpers/LLFlow).
        """
        return Path(__file__).parent.parent / "helpers" / "LLFlow"

    def _llflow_sync_weights(self, root: Path) -> Path:
        """
        weight_and_models içindeki hazır ağırlıkları beklenen konuma kopyalar.
        - Model: LLFlow/LOLv2.pth
        
        Returns:
            Model ağırlık dosyasının yolu
        """
        weights_src_dir = Path(__file__).parent.parent / "weight_and_models" / "LLFlow"
        
        # Model dosyasını bul (.pth uzantılı)
        model_files = list(weights_src_dir.glob("*.pth"))
        if not model_files:
            raise ValueError(f"LLFlow model ağırlığı bulunamadı: {weights_src_dir}")
        
        # İlk .pth dosyasını kullan (veya en son olanı)
        model_src = max(model_files, key=lambda p: p.stat().st_mtime)
        
        # LLFlow root klasörüne kopyala
        root.mkdir(parents=True, exist_ok=True)
        
        model_dst = root / model_src.name
        if not model_dst.exists():
            shutil.copy(model_src, model_dst)
            logger.info("LLFlow ağırlığı kopyalandı: %s", model_dst)
        else:
            logger.info("LLFlow ağırlığı zaten mevcut: %s", model_dst)
        
        return model_dst

    def _llflow_imread(self, path: Path):
        """Görüntüyü RGB formatında okur"""
        img = cv2.imread(str(path))
        if img is None:
            raise ValueError(f"Görüntü okunamadı: {path}")
        return img[:, :, [2, 1, 0]]  # BGR'den RGB'ye çevir

    def _llflow_imwrite(self, path: Path, img):
        """Görüntüyü kaydeder"""
        path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(path), img[:, :, [2, 1, 0]])  # RGB'den BGR'ye çevir

    def _llflow_t(self, array):
        """NumPy array'i PyTorch tensor'üne çevirir"""
        return torch.Tensor(np.expand_dims(array.transpose([2, 0, 1]), axis=0).astype(np.float32)) / 255

    def _llflow_rgb(self, tensor):
        """PyTorch tensor'ünü RGB görüntüye çevirir"""
        return (np.clip((tensor[0] if len(tensor.shape) == 4 else tensor).detach().cpu().numpy().transpose([1, 2, 0]), 0, 1) * 255).astype(np.uint8)

    def _llflow_auto_padding(self, img, times=16):
        """Görüntüyü belirli bir sayıya bölünebilir hale getirmek için padding yapar"""
        h, w, _ = img.shape
        h1, w1 = (times - h % times) // 2, (times - w % times) // 2
        h2, w2 = (times - h % times) - h1, (times - w % times) - w1
        img = cv2.copyMakeBorder(img, h1, h2, w1, w2, cv2.BORDER_REFLECT)
        return img, [h1, h2, w1, w2]

    def _llflow_hiseq_color_cv2_img(self, img):
        """Histogram eşitleme yapar"""
        # RGB'den BGR'ye çevir (cv2 fonksiyonları için)
        img_bgr = img[:, :, [2, 1, 0]]
        (b, g, r) = cv2.split(img_bgr)
        bH = cv2.equalizeHist(b)
        gH = cv2.equalizeHist(g)
        rH = cv2.equalizeHist(r)
        result = cv2.merge((bH, gH, rH))
        # BGR'den RGB'ye geri çevir
        return result[:, :, [2, 1, 0]]

    def _llflow_get_default_config(self):
        """Varsayılan config dictionary'si döndürür"""
        return {
            'name': 'independent_test',
            'use_tb_logger': False,
            'model': 'LLFlow',
            'distortion': 'sr',
            'scale': 1,
            'gpu_ids': [],
            'dataset': 'LoL',
            'optimize_all_z': False,
            'cond_encoder': 'ConEncoder1',
            'train_gt_ratio': 0.5,
            'avg_color_map': False,
            'concat_histeq': True,
            'histeq_as_input': False,
            'concat_color_map': False,
            'gray_map': False,
            'align_condition_feature': False,
            'align_weight': 0.001,
            'align_maxpool': True,
            'to_yuv': False,
            'encode_color_map': False,
            'le_curve': False,
            'datasets': {
                'train': {
                    'root': '',
                    'quant': 32,
                    'use_shuffle': True,
                    'n_workers': 1,
                    'batch_size': 16,
                    'use_flip': True,
                    'color': 'RGB',
                    'use_crop': True,
                    'GT_size': 160,
                    'noise_prob': 0,
                    'noise_level': 5,
                    'log_low': True,
                    'gamma_aug': False
                },
                'val': {
                    'root': '',
                    'n_workers': 1,
                    'quant': 32,
                    'n_max': 20,
                    'batch_size': 1,
                    'log_low': True
                }
            },
            'heat': 0,
            'network_G': {
                'which_model_G': 'LLFlow',
                'in_nc': 3,
                'out_nc': 3,
                'nf': 64,
                'nb': 24,
                'train_RRDB': False,
                'train_RRDB_delay': 0.5,
                'flow': {
                    'K': 12,
                    'L': 3,
                    'noInitialInj': True,
                    'coupling': 'CondAffineSeparatedAndCond',
                    'additionalFlowNoAffine': 2,
                    'split': {
                        'enable': False
                    },
                    'fea_up0': True,
                    'stackRRDB': {
                        'blocks': [1, 3, 5, 7],
                        'concat': True
                    }
                }
            },
            'path': {
                'strict_load': True,
                'resume_state': 'auto'
            },
            'train': {
                'manual_seed': 10,
                'lr_G': 5e-4,
                'weight_decay_G': 0,
                'beta1': 0.9,
                'beta2': 0.99,
                'lr_scheme': 'MultiStepLR',
                'warmup_iter': 200,
                'lr_steps_rel': [0.5, 0.75, 0.9, 0.95],
                'lr_gamma': 0.5,
                'weight_l1': 0,
                'weight_fl': 1,
                'niter': 40000,
                'val_freq': 1000
            },
            'val': {
                'n_sample': 4
            },
            'test': {
                'heats': [0.0, 0.7, 0.8, 0.9]
            },
            'logger': {
                'print_freq': 100,
                'save_checkpoint_freq': 1e4
            }
        }

    def _llflow_load_model(self, root: Path, model_path: Path, config_path: Optional[Path] = None):
        """Modeli config dosyasından veya varsayılan config'den yükler"""
        # code klasörünü path'e ekle
        code_dir = root / "code"
        code_dir_str = str(code_dir)
        if code_dir_str not in sys.path:
            sys.path.insert(0, code_dir_str)

        import options.options as option
        from models import create_model

        if config_path and config_path.exists():
            # Config dosyasından yükle
            opt = option.parse(str(config_path), is_train=False)
        else:
            # Varsayılan config kullan
            logger.info("LLFlow: Config dosyası belirtilmedi veya bulunamadı, varsayılan config kullanılıyor...")
            default_config = self._llflow_get_default_config()
            
            # Geçici bir YAML dosyası oluştur
            with tempfile.NamedTemporaryFile(mode='w', suffix='.yml', delete=False) as f:
                yaml.dump(default_config, f, default_flow_style=False, allow_unicode=True)
                temp_config_path = f.name
            
            try:
                opt = option.parse(temp_config_path, is_train=False)
            finally:
                # Geçici dosyayı sil
                os.unlink(temp_config_path)
        
        # options.py parse işlemi için gpu_ids boş liste olmalı (None kabul etmiyor)
        # Ama BaseModel None bekliyor, bu yüzden parse sonrası None yapacağız
        opt = option.dict_to_nonedict(opt)
        
        # BaseModel için gpu_ids None olmalı (CPU modu)
        # BaseModel: self.device = torch.device('cuda' if opt.get('gpu_ids', None) is not None else 'cpu')
        if not opt.get('gpu_ids') or len(opt.get('gpu_ids', [])) == 0:
            opt['gpu_ids'] = None
        
        # Model yolunu override et
        opt['model_path'] = str(model_path)
        
        model = create_model(opt)
        model.load_network(load_path=str(model_path), network=model.netG)
        
        # BaseModel zaten device'ı ayarlıyor, ekstra taşıma gerekmez
        # Ancak CUDA yoksa CPU'ya taşımayı garanti et
        if not torch.cuda.is_available() and model.device.type == 'cuda':
            model.device = torch.device('cpu')
            model.netG = model.netG.cpu()
        
        return model, opt

    def _run_llflow(self, image_bytes: bytes, original_filename: str) -> bytes:
        """
        LLFlow test/inference çalıştırır, çıktı bytes döner.
        test_unpaired.py'deki fonksiyonları kullanır.
        """
        root = self._get_llflow_root()
        if not root.exists():
            raise ValueError(f"LLFlow kökü bulunamadı: {root}")

        # code klasörünü path'e ekle
        code_dir = root / "code"
        code_dir_str = str(code_dir)
        if code_dir_str not in sys.path:
            sys.path.insert(0, code_dir_str)

        # test_unpaired.py'deki fonksiyonları import et
        try:
            import test_unpaired as llflow_test
        except ImportError as e:
            raise ValueError(f"LLFlow test_unpaired.py import edilemedi: {e}")

        # Geçici dosyalar için klasör oluştur
        suffix = Path(original_filename).suffix or ".jpg"
        tmp_dir = Path(tempfile.mkdtemp(prefix="llflow_", dir=root))
        input_path = tmp_dir / f"input{suffix}"
        output_path = tmp_dir / "output.jpg"
        
        # Input görseli kaydet
        input_path.write_bytes(image_bytes)

        try:
            # Ağırlıkları senkronize et
            model_path = self._llflow_sync_weights(root)

            # Config dosyası bul (varsa) veya varsayılan config kullan
            config_path = root / "code" / "confs" / "LOLv2-pc.yml"
            if not config_path.exists():
                config_path = None  # Varsayılan config kullanılacak
            
            # Modeli yükle (test_unpaired.py'deki load_model kullan)
            logger.info(f"LLFlow: Model yükleniyor: {model_path}")
            if config_path and config_path.exists():
                # Config dosyasından model yükle
                # Önce config'i parse et ve model_path'i ekle
                import options.options as option
                from utils.util import opt_get
                temp_opt = option.parse(str(config_path), is_train=False)
                temp_opt['gpu_ids'] = None
                temp_opt['model_path'] = str(model_path)  # Model path'i ekle
                temp_opt = option.dict_to_nonedict(temp_opt)
                
                # Geçici bir YAML dosyası oluştur ve model_path'i ekle
                import yaml
                with open(config_path, 'r') as f:
                    config_dict = yaml.load(f, Loader=yaml.FullLoader)
                config_dict['model_path'] = str(model_path)
                
                with tempfile.NamedTemporaryFile(mode='w', suffix='.yml', delete=False) as f:
                    yaml.dump(config_dict, f, default_flow_style=False, allow_unicode=True)
                    temp_config_path = f.name
                
                try:
                    model, opt = llflow_test.load_model(temp_config_path)
                finally:
                    os.unlink(temp_config_path)
            else:
                # Varsayılan config ile model yükle
                model, opt = self._llflow_load_model(root, model_path, config_path=None)
            
            # CPU modunu garanti et
            if not torch.cuda.is_available():
                model.netG = model.netG.cpu()
                model.device = torch.device('cpu')

            # Görüntüyü oku (test_unpaired.py'deki imread kullan)
            logger.info(f"LLFlow: Görüntü okunuyor: {input_path}")
            lr = llflow_test.imread(str(input_path))
            raw_shape = lr.shape
            
            # Padding yap (test_unpaired.py'deki auto_padding kullan)
            lr, padding_params = llflow_test.auto_padding(lr)
            
            # Histogram eşitleme (test_unpaired.py'deki hiseq_color_cv2_img kullan)
            his = llflow_test.hiseq_color_cv2_img(lr)
            if opt.get("histeq_as_input", False):
                lr = his
            
            # Tensor'e çevir (test_unpaired.py'deki t kullan)
            lr_t = llflow_test.t(lr)
            
            # Log transformasyonu (eğer config'de varsa)
            if opt["datasets"]["train"].get("log_low", False):
                lr_t = torch.log(torch.clamp(lr_t + 1e-3, min=1e-3))
            
            # Histogram eşitleme concatenation (eğer config'de varsa)
            if opt.get("concat_histeq", False):
                his_t = llflow_test.t(his)
                lr_t = torch.cat([lr_t, his_t], dim=1)
            
            # Model inference
            logger.info("LLFlow: Model çalıştırılıyor...")
            device = next(model.netG.parameters()).device
            
            lr_t = lr_t.to(device)
            
            if torch.cuda.is_available() and device.type == 'cuda':
                with torch.cuda.amp.autocast():
                    sr_t = model.get_sr(lq=lr_t, heat=None)
            else:
                with torch.no_grad():
                    sr_t = model.get_sr(lq=lr_t, heat=None)
            
            # Padding'i kaldır ve RGB'ye çevir (test_unpaired.py'deki rgb kullan)
            sr = llflow_test.rgb(torch.clamp(sr_t, 0, 1)[:, :, padding_params[0]:sr_t.shape[2] - padding_params[1],
                         padding_params[2]:sr_t.shape[3] - padding_params[3]])
            
            # Orijinal boyut kontrolü
            assert raw_shape == sr.shape, f"Boyut uyuşmazlığı: {raw_shape} != {sr.shape}"
            
            # Sonucu kaydet (test_unpaired.py'deki imwrite kullan)
            logger.info(f"LLFlow: Sonuç kaydediliyor: {output_path}")
            llflow_test.imwrite(str(output_path), sr)
            
            # Çıktıyı oku
            output_bytes = output_path.read_bytes()
            
            return output_bytes

        except Exception as exc:
            logger.exception("LLFlow predict başarısız")
            raise ValueError(f"LLFlow predict başarısız: {exc}") from exc
        finally:
            # Temizlik
            try:
                shutil.rmtree(tmp_dir, ignore_errors=True)
            except Exception:
                pass

