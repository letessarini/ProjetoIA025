
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

# ============================================
# CONFIGURAÇÕES
# ============================================

PATCH = Path(
    "/home/leticia/ProjetoIA025/patches/"
    "20130107_y0512_x0256.npz"
)

OUTPUT = Path(
    "/home/leticia/ProjetoIA025/visualizacao_patch.png"
)

NOMES_CANAIS = [
    "Precipitação total",
    "Precipitação máxima",
    "Uso do solo",
    "Impermeabilização",
    "Declividade",
    "Suscetibilidade das vias",
    "HAND",
    "Área de contribuição (log)",
    "Altitude"
]

# ============================================
# CARREGAR ARQUIVO
# ============================================

with np.load(PATCH, allow_pickle=False) as dados:
    print("\nCampos disponíveis:")
    for chave in dados.files:
        valor = dados[chave]
        print(f"  {chave}: shape={valor.shape}, dtype={valor.dtype}")

    X = dados["X"].astype(np.float32)
    y = dados["y"]
    valido = dados["valido"]

    data = dados["date"].item()
    precip_val = dados["precip_val"].item()
    frac_valida = dados["frac_valida"].item()

print("\nInformações do patch:")
print("Data:", data)
print("Precipitação:", precip_val)
print("Fração válida:", frac_valida)
print("Pixels inundados:", int((y == 1).sum()))
print(
    "Pixels inundados válidos:",
    int(((y == 1) & (valido == 1)).sum())
)

# ============================================
# VISUALIZAÇÃO
# ============================================

fig, axes = plt.subplots(3, 4, figsize=(17, 12))
axes = axes.flatten()

# Visualizar apenas os 9 canais físicos
for i in range(9):
    ax = axes[i]

    im = ax.imshow(X[i], cmap="viridis")
    ax.set_title(
        f"Canal {i}: {NOMES_CANAIS[i]}",
        fontsize=10,
        pad=8
    )
    ax.axis("off")

    plt.colorbar(
        im,
        ax=ax,
        fraction=0.046,
        pad=0.04
    )

# Máscara de inundação
axes[9].imshow(y, cmap="Blues", vmin=0, vmax=1)
axes[9].set_title(
    "Máscara de inundação (y)",
    fontsize=10,
    pad=8
)
axes[9].axis("off")

# Ocultar os dois espaços restantes
axes[10].axis("off")
axes[11].axis("off")

# Título principal
fig.suptitle(
    f"Patch: {PATCH.name}\n"
    f"Data: {data} | Precipitação: {precip_val:.2f} mm",
    fontsize=11,
    y=0.98
)

# Ajustar espaçamento
plt.tight_layout(rect=[0, 0, 1, 0.92], h_pad=2.0, w_pad=1.5)

# Salvar imagem
plt.savefig(OUTPUT, dpi=150, bbox_inches="tight")
plt.show()

print(f"\nVisualização salva em: {OUTPUT}")
