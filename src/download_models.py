from pathlib import Path
import requests

MDX_DOWNLOAD_LINK = 'https://github.com/TRvlvr/model_repo/releases/download/all_public_uvr_models/'
RVC_DOWNLOAD_LINK = 'https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/'

BASE_DIR = Path(__file__).resolve().parent.parent
mdxnet_models_dir = BASE_DIR / 'mdxnet_models'
rvc_models_dir = BASE_DIR / 'rvc_models'


def dl_model(link, model_name, dir_name):
    with requests.get(f'{link}{model_name}') as r:
        r.raise_for_status()
        with open(dir_name / model_name, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)


MDX_PARAMS_LINK = 'https://raw.githubusercontent.com/TRvlvr/application_data/main/mdx_model_data/model_data.json'


if __name__ == '__main__':
    mdxnet_models_dir.mkdir(exist_ok=True)
    rvc_models_dir.mkdir(exist_ok=True)

    # model_data.json: thông số của các model MDX (cần cho bước tách vocal)
    if not (mdxnet_models_dir / 'model_data.json').exists():
        print('Downloading model_data.json...')
        try:
            with requests.get(MDX_PARAMS_LINK, timeout=60) as r:
                r.raise_for_status()
                (mdxnet_models_dir / 'model_data.json').write_bytes(r.content)
        except Exception as e:
            print(f'[!] Không tải được model_data.json ({e}). Hãy tự đặt file này vào mdxnet_models/.')

    mdx_model_names = ['UVR-MDX-NET-Voc_FT.onnx', 'UVR_MDXNET_KARA_2.onnx', 'Reverb_HQ_By_FoxJoy.onnx']
    for model in mdx_model_names:
        if (mdxnet_models_dir / model).exists():
            continue
        print(f'Downloading {model}...')
        dl_model(MDX_DOWNLOAD_LINK, model, mdxnet_models_dir)

    rvc_model_names = ['hubert_base.pt', 'rmvpe.pt']
    for model in rvc_model_names:
        if (rvc_models_dir / model).exists():
            continue
        print(f'Downloading {model}...')
        dl_model(RVC_DOWNLOAD_LINK, model, rvc_models_dir)

    print('All models downloaded!')
