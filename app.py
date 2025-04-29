# Lung CT Web App – CT Upload Only + Metrics + Chatbot + Ground Truth + Prediction

import streamlit as st
import torch
import torch.nn as nn
import timm
import numpy as np
from torchvision import transforms
from PIL import Image
from summary_generator import generate_summary
from qa_engine import generate_response
from hospital_lookup import get_hospitals_by_pincode

st.set_page_config(page_title="🫁 Lung CT Segmentation & Assistant", layout="wide")
st.markdown("""
<style>
.big-title {text-align: center; background-color: #1565c0; color: white; padding: 1rem; font-size: 28px; border-radius: 10px;}
.footer {text-align: center; font-size: 13px; color: gray; margin-top: 40px;}
</style>
""", unsafe_allow_html=True)
st.markdown("<div class='big-title'>🫁 Lung CT Segmentation and Metrics Assistant</div>", unsafe_allow_html=True)

class AdaptiveEdgeAttention(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, 1, 3, padding=1)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        return x * self.sigmoid(self.conv(x))

class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.ReLU()
        )
    def forward(self, x):
        return self.block(x)

class MobileViT_QRC_U_Net(nn.Module):
    def __init__(self, in_channels=1, out_channels=1):
        super().__init__()
        self.encoder = timm.create_model('mobilevit_xxs', pretrained=False, features_only=True, in_chans=in_channels)
        chs = self.encoder.feature_info.channels()
        self.bridge = AdaptiveEdgeAttention(chs[-1])
        self.up4 = nn.ConvTranspose2d(chs[-1], chs[-2], 2, 2)
        self.dec4 = DecoderBlock(chs[-2]*2, chs[-2])
        self.up3 = nn.ConvTranspose2d(chs[-2], chs[-3], 2, 2)
        self.dec3 = DecoderBlock(chs[-3]*2, chs[-3])
        self.up2 = nn.ConvTranspose2d(chs[-3], chs[-4], 2, 2)
        self.dec2 = DecoderBlock(chs[-4]*2, chs[-4])
        self.up1 = nn.ConvTranspose2d(chs[-4], chs[-4]//2, 2, 2)
        self.dec1 = DecoderBlock(chs[-4]//2, chs[-4]//2)
        self.out_conv = nn.Conv2d(chs[-4]//2, out_channels, 1)
        self.upsample = nn.Upsample(size=(256, 256), mode='bilinear', align_corners=True)

    def forward(self, x):
        e1, e2, e3, e4, e5 = self.encoder(x)
        x = self.bridge(e5)
        x = self.dec4(torch.cat([self.up4(x), e4], 1))
        x = self.dec3(torch.cat([self.up3(x), e3], 1))
        x = self.dec2(torch.cat([self.up2(x), e2], 1))
        x = self.dec1(self.up1(x))
        x = self.out_conv(x)
        return torch.sigmoid(self.upsample(x))

@st.cache_resource
def load_model():
    model = MobileViT_QRC_U_Net()
    state_dict = torch.load("best_mobilevit_qrc_unet.pth", map_location="cpu")
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    return model

model = load_model()

# Upload CT scan (with GT mask as second channel)
st.header("📤 Upload Lung CT Image")
ct_img = st.file_uploader("Upload CT scan image (single image where Red=GT, Green=CT)", type=["png", "jpg", "jpeg"])

if ct_img:
    color_img = Image.open(ct_img).convert("RGB").resize((256, 256))
    ct_gray = color_img.getchannel("G").convert("L")  # Green channel = original image
    gt_mask = color_img.getchannel("R").convert("L")  # Red channel = GT mask

    tensor = transforms.ToTensor()(ct_gray).unsqueeze(0)
    gt_np = np.array(gt_mask.resize((256, 256)))
    gt_bin = (gt_np > 128).astype(np.uint8)

    with torch.no_grad():
        pred = model(tensor).squeeze().numpy()
        pred_bin = (pred > 0.5).astype(np.uint8)

    pred_img = Image.fromarray(pred_bin * 255)
    overlay = np.array(ct_gray.convert("RGB"))
    overlay[pred_bin > 0] = [255, 0, 0]  # Highlight prediction

    intersection = np.logical_and(pred_bin, gt_bin).sum()
    union = np.logical_or(pred_bin, gt_bin).sum()
    iou = intersection / (union + 1e-6)
    dice = (2 * intersection) / (pred_bin.sum() + gt_bin.sum() + 1e-6)
    confidence = float((pred > 0.75).mean() * 100)

    st.subheader("🖼️ Results")
    c1, c2, c3 = st.columns(3)
    c1.image(ct_gray, caption="Original CT Scan", use_column_width=True)
    c2.image(gt_mask, caption="Ground Truth Mask", use_column_width=True)
    c3.image(pred_img, caption="Predicted Mask", use_column_width=True)
    st.image(overlay, caption="Overlay", use_column_width=True)

    st.subheader("📊 Metrics")
    st.markdown(f"- **Confidence Score**: {confidence:.2f}%")
    st.markdown(f"- **IoU Score**: {iou:.4f}")
    st.markdown(f"- **Dice Score**: {dice:.4f}")

    features = {
        "is_malignant": confidence > 80,
        "confidence_score": confidence,
        "iou_score": iou,
        "dice_score": dice
    }

    st.subheader("📝 AI Summary")
    st.success(generate_summary(features))

    st.subheader("💬 Ask a Question")
    user_q = st.text_input("Type your question about this CT scan result:")
    if user_q:
        st.info(generate_response(user_q, features))

st.markdown("<div class='footer'>Built by Team 2 • QRC-U-Net • Streamlit • PyTorch</div>", unsafe_allow_html=True)
