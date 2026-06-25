"""
实时验证 Web 应用 (Flask)
=========================
功能：实时显示验证图片、预测结果、32个滤波器特征图、
      动态准确率、近50次趋势、图片上传验证
运行: python app.py
访问: http://localhost:5000
"""
import torch
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import numpy as np
import base64
import io
import json
from PIL import Image, ImageOps
from flask import Flask, render_template, request, jsonify
from collections import deque

from model import STDP_ConvLayer, MLP_Classifier, poisson_encode
from train import predict_with_confidence, load_best_model, load_features

app = Flask(__name__)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'设备: {device}')

# ── 加载模型 ────────────────────────────────────────────
print('加载 STDP 模型...')
conv_layer = STDP_ConvLayer(n_filters=32, T=10).to(device)
ckpt = torch.load('checkpoints/checkpoint.pth', map_location=device, weights_only=False)
if ckpt.get('stdp_weights'):
    conv_layer.load_state_dict(ckpt['stdp_weights'])

print('加载最佳 MLP 模型...')
classifier = MLP_Classifier(hidden_dim=512, n_classes=11, dropout=0.3).to(device)
classifier, best_acc = load_best_model(classifier, device)
classifier.eval()
conv_layer.eval()
print(f'模型加载完成 (MLP 最佳 acc={best_acc:.4f})')

# ── 加载测试集 ──────────────────────────────────────────
transform = transforms.ToTensor()
test_set = torchvision.datasets.MNIST(root='./data', train=False, transform=transform, download=False)
print(f'测试集: {len(test_set)} 张')

# ── 验证状态 ────────────────────────────────────────────
CONFIDENCE_THRESHOLD = 0.3
state = {
    'current_idx': 0,
    'total_tested': 0,
    'correct': 0,
    'correct_no_undef': 0,  # 不含 undefined 的正确数
    'undefined_count': 0,
    'recent_50': deque(maxlen=50),  # [{'pred': int, 'true': int, 'correct': bool, 'conf': float}]
    'shuffled_indices': np.random.RandomState(42).permutation(len(test_set)),
}


def reset_state():
    state['current_idx'] = 0
    state['total_tested'] = 0
    state['correct'] = 0
    state['correct_no_undef'] = 0
    state['undefined_count'] = 0
    state['recent_50'].clear()
    state['shuffled_indices'] = np.random.RandomState(42).permutation(len(test_set))


def tensor_to_b64(tensor, cmap='gray', size=None):
    """Tensor → base64 PNG"""
    arr = tensor.detach().cpu().numpy()
    if arr.ndim == 3 and arr.shape[0] == 1:
        arr = arr[0]
    if cmap == 'gray':
        arr = (arr - arr.min()) / (arr.max() - arr.min() + 1e-8) * 255
        img = Image.fromarray(arr.astype(np.uint8), mode='L')
    elif cmap == 'RdBu':
        arr = (arr - arr.min()) / (arr.max() - arr.min() + 1e-8)
        arr = (arr * 255).astype(np.uint8)
        img = Image.fromarray(arr, mode='L')
    else:
        arr = (arr - arr.min()) / (arr.max() - arr.min() + 1e-8) * 255
        img = Image.fromarray(arr.astype(np.uint8))
    if size:
        img = img.resize((size, size), Image.NEAREST)
    buf = io.BytesIO()
    img.save(buf, format='PNG')
    return base64.b64encode(buf.getvalue()).decode()


def get_feature_maps(image_tensor):
    """获取 32 个滤波器的特征图 (28×28，未池化)"""
    conv_layer.eval()
    with torch.no_grad():
        img = image_tensor.unsqueeze(0).to(device)  # [1, 1, 28, 28]
        spikes = poisson_encode(img, T=10)           # [10, 1, 1, 28, 28]
        conv_layer.if_node.reset()
        all_spikes = []
        for t in range(10):
            out = conv_layer.conv(spikes[t])
            out = conv_layer.if_node(out)
            all_spikes.append(out)
        rates = torch.stack(all_spikes).mean(0).squeeze(0)  # [32, 28, 28]
        return rates.cpu()


# ── 路由 ────────────────────────────────────────────────

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/api/next')
def api_next():
    """获取下一个验证样本的结果"""
    if state['current_idx'] >= len(test_set):
        return jsonify({'done': True, 'message': '全部测试样本已遍历完毕'})

    idx = state['shuffled_indices'][state['current_idx']]
    img, true_label = test_set[idx]

    # 前向推理
    conv_layer.eval()
    with torch.no_grad():
        img_t = img.unsqueeze(0).to(device)
        features = conv_layer.extract_features(img_t, T=10)
        pred, conf = predict_with_confidence(
            classifier, features.to(device), device=device, threshold=CONFIDENCE_THRESHOLD
        )
        pred = pred[0].item()
        conf = conf[0].item()

    is_correct = (pred == true_label)
    is_undefined = (pred == 10)

    # 更新统计
    state['total_tested'] += 1
    if is_correct:
        state['correct'] += 1
    if is_correct and not is_undefined:
        state['correct_no_undef'] += 1
    if is_undefined:
        state['undefined_count'] += 1

    state['recent_50'].append({
        'pred': pred, 'true': int(true_label), 'correct': is_correct, 'conf': round(conf, 3)
    })
    state['current_idx'] += 1

    # 获取原始图像 B64
    img_b64 = tensor_to_b64(img, cmap='gray', size=140)

    # 获取 32 个特征图
    fmaps = get_feature_maps(img)
    # 拼成 4×8 网格
    fmaps_grid = []
    for r in range(4):
        row = []
        for c in range(8):
            f_idx = r * 8 + c
            fm = fmaps[f_idx]
            row.append(tensor_to_b64(fm, cmap='RdBu', size=70))
        fmaps_grid.append(row)

    avg_acc = state['correct'] / state['total_tested'] if state['total_tested'] > 0 else 0

    # 近50次趋势数据
    recent_data = [
        {'correct': item['correct'], 'pred': item['pred'], 'true': item['true']}
        for item in state['recent_50']
    ]

    return jsonify({
        'done': False,
        'image_b64': img_b64,
        'true_label': int(true_label),
        'pred_label': pred if pred != 10 else '?',
        'confidence': round(conf, 3),
        'is_correct': is_correct,
        'is_undefined': is_undefined,
        'feature_maps': fmaps_grid,
        'total_tested': state['total_tested'],
        'correct': state['correct'],
        'avg_accuracy': round(avg_acc, 4),
        'undefined_count': state['undefined_count'],
        'recent_50': recent_data,
        'progress': state['current_idx'],
        'total': len(test_set),
    })


@app.route('/api/stats')
def api_stats():
    """获取当前统计信息（不推进进度）"""
    avg_acc = state['correct'] / state['total_tested'] if state['total_tested'] > 0 else 0
    return jsonify({
        'total_tested': state['total_tested'],
        'correct': state['correct'],
        'avg_accuracy': round(avg_acc, 4),
        'undefined_count': state['undefined_count'],
        'progress': state['current_idx'],
        'total': len(test_set),
    })


@app.route('/api/reset')
def api_reset():
    """重置验证状态"""
    reset_state()
    return jsonify({'status': 'ok', 'message': '已重置'})


def preprocess_upload_image(file):
    """
    预处理 → 强制黑底白字 28×28（与 MNIST 训练数据完全一致）。
    无论输入是截图、拍照、白底黑字、黑底白字，都统一转换。
    """
    img = Image.open(file).convert('L')
    img = img.resize((28, 28), Image.LANCZOS)
    arr = np.array(img, dtype=np.float32)

    # 判断当前是白底黑字还是黑底白字：
    # 取四角（背景区域）像素平均值
    corners = np.concatenate([
        arr[0, :], arr[-1, :], arr[1:-1, 0], arr[1:-1, -1]
    ])
    bg_is_light = corners.mean() > 128  # 背景偏亮 → 白底

    if bg_is_light:
        # 白底 → 反色 → 黑底
        arr = 255 - arr

    # 二值化 + 归一化：黑底(0)白字(1)
    # 用 Otsu 近似：以均值为阈值
    thresh = arr.mean()
    arr = (arr > thresh).astype(np.float32)

    # 确保至少有一些白像素（否则是全黑图 → 模型无法识别）
    if arr.sum() < 5:
        # 太稀疏，尝试降低阈值
        arr = np.array(Image.open(file).convert('L').resize((28, 28), Image.LANCZOS), dtype=np.float32)
        if bg_is_light:
            arr = 255 - arr
        arr = (arr > arr.mean() * 0.5).astype(np.float32)

    img_t = torch.tensor(arr, dtype=torch.float32).unsqueeze(0)  # [1, 28, 28]
    return img_t


@app.route('/api/upload', methods=['POST'])
def api_upload():
    """上传图片并预测"""
    if 'image' not in request.files:
        return jsonify({'error': '未找到图片'}), 400

    file = request.files['image']
    try:
        img_t = preprocess_upload_image(file).to(device)

        # 前向推理
        conv_layer.eval()
        with torch.no_grad():
            img_4d = img_t.unsqueeze(0)  # [1, 1, 28, 28]
            features = conv_layer.extract_features(img_4d, T=10)
            pred, conf = predict_with_confidence(
                classifier, features.to(device), device=device, threshold=CONFIDENCE_THRESHOLD
            )

        label_name = 'UNDEFINED' if pred[0].item() == 10 else str(pred[0].item())

        # 预处理后的图片 B64（供调试查看）
        import base64, io as _io
        arr_debug = (img_t.squeeze().cpu().numpy() * 255).astype(np.uint8)
        debug_img = Image.fromarray(arr_debug, mode='L')
        buf = _io.BytesIO()
        debug_img.resize((140, 140), Image.NEAREST).save(buf, format='PNG')
        debug_b64 = base64.b64encode(buf.getvalue()).decode()

        return jsonify({
            'prediction': label_name,
            'class_id': int(pred[0].item()),
            'confidence': round(conf[0].item(), 3),
            'is_undefined': pred[0].item() == 10,
            'debug_image': f'data:image/png;base64,{debug_b64}',
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 400


@app.route('/api/skip', methods=['POST'])
def api_skip():
    """跳转到指定数量的样本之后（用于快速跳过）"""
    n = request.json.get('n', 100)
    state['current_idx'] = min(state['current_idx'] + n, len(test_set))
    # 跳过时不更新统计
    return jsonify({'status': 'ok', 'new_idx': state['current_idx']})


if __name__ == '__main__':
    import os
    os.makedirs('templates', exist_ok=True)
    print('启动服务器: http://localhost:5000')
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
