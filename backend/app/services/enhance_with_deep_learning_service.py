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
from typing import Callable, Dict
import logging
from PIL import Image

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

