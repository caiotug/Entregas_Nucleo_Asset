"""
===============================================================
ANÁLISE DE PORTFÓLIO DE MARKOWITZ
Ações: ITUB4, WEGE3, BRSR6, CBAV3, FIQE3
===============================================================

CONCEITO:
  Harry Markowitz (1952) mostrou que diversificar ativos com
  correlações diferentes reduz o risco total do portfólio sem
  necessariamente reduzir o retorno esperado.

  O objetivo é encontrar a "Fronteira Eficiente" — o conjunto de
  portfólios que maximizam o retorno para cada nível de risco.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.cm as cm
import yfinance as yf
import warnings

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────
# PASSO 1: BAIXAR OS DADOS HISTÓRICOS
# ─────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("  PASSO 1: Baixando dados históricos das ações")
print("="*60)

TICKERS = {
    "ITUB4": "Itaú Unibanco (Banco)",
    "WEGE3": "WEG (Equipamentos Elétricos)",
    "BRSR6": "Banrisul (Banco Regional)",
    "CBAV3": "CBA - Companhia Brasileira de Alumínio (Mineração e Alumínio)",
    "FIQE3": "Unifique (Telecom / Internet Fibra)",
}

tickers_br = [t + ".SA" for t in TICKERS.keys()]  # Yahoo usa sufixo .SA para B3

print(f"\nBaixando 2 anos de preços de fechamento ajustados...")

precos = yf.download(
    tickers_br,
    period="2y",
    auto_adjust=True,
    progress=False,
)["Close"]

precos.columns = list(TICKERS.keys())  # renomear para nomes simples
precos.dropna(how="any", inplace=True)

print(f"\n✓ Dados obtidos: {len(precos)} pregões de {precos.index[0].date()} a {precos.index[-1].date()}")
print(precos.tail(3).round(2))


# ─────────────────────────────────────────────────────────────
# PASSO 2: CALCULAR RETORNOS LOGARÍTMICOS DIÁRIOS
# ─────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("  PASSO 2: Calculando retornos diários")
print("="*60)

"""
Por que retorno logarítmico?
  r_log = ln(P_t / P_{t-1})
  
  Vantagens:
  - Pode ser somado no tempo (2 dias = r_dia1 + r_dia2)
  - Simetria: queda de 50% e alta de 100% equivalem
  - Melhor aproximação para distribuição normal
"""

retornos = np.log(precos / precos.shift(1)).dropna()

print("\nEstatísticas dos retornos diários (%):")
stats = retornos.describe() * 100
print(stats.round(3))


# ─────────────────────────────────────────────────────────────
# PASSO 3: PARÂMETROS DE MARKOWITZ
# ─────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("  PASSO 3: Calculando parâmetros do modelo")
print("="*60)

DIAS_UTEIS = 252  # dias de negociação por ano no Brasil

# Retorno médio anualizado de cada ação
retorno_medio = retornos.mean() * DIAS_UTEIS

# Matriz de covariância anualizada
# Mede como os ativos se movem JUNTOS — chave da diversificação
cov_matrix = retornos.cov() * DIAS_UTEIS

# Matriz de correlação (para interpretação humana)
corr_matrix = retornos.corr()

print("\nRetorno anual esperado por ação:")
for ticker, ret in retorno_medio.items():
    print(f"  {ticker:6s}: {ret*100:+.1f}%")

print("\nMatriz de Correlação (quanto cada par se move junto):")
print(corr_matrix.round(3))
print("\n  Valores próximos de +1 = movem-se juntos (pouca diversificação)")
print("  Valores próximos de -1 = movem-se opostos (ótima diversificação)")
print("  Valores próximos de  0 = movem-se independentemente")


# ─────────────────────────────────────────────────────────────
# PASSO 4: SIMULAÇÃO DE MONTE CARLO
# ─────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("  PASSO 4: Simulando 20.000 portfólios aleatórios")
print("="*60)

"""
Ideia: gerar milhares de combinações aleatórias de pesos (w1, w2, ..., wN)
onde w1 + w2 + ... + wN = 1 e cada wi >= 0 (sem venda a descoberto).

Para cada combinação calcular:
  Retorno do portfólio: E[Rp] = Σ wi * E[Ri]
  Risco do portfólio:   σp = √(w' · Σ · w)
    onde Σ é a matriz de covariância

O "milagre" de Markowitz: σp < Σ wi * σi  (quando correlações < 1)
"""

N_SIM = 20_000
N_ATIVOS = len(TICKERS)

# Arrays para armazenar resultados
port_retornos  = np.zeros(N_SIM)
port_riscos    = np.zeros(N_SIM)
port_sharpes   = np.zeros(N_SIM)
port_pesos     = np.zeros((N_SIM, N_ATIVOS))

SELIC_ANUAL = 0.1075  # Taxa Selic (~10,75% ao ano como referência)

print(f"\nReferência livre de risco (Selic): {SELIC_ANUAL*100:.2f}% a.a.")

rng = np.random.default_rng(seed=42)

for i in range(N_SIM):
    # Gerar pesos aleatórios que somam 1
    pesos = rng.random(N_ATIVOS)
    pesos /= pesos.sum()

    # Retorno esperado do portfólio
    ret = np.dot(pesos, retorno_medio.values)

    # Variância do portfólio (a fórmula matricial captura as correlações)
    var = pesos @ cov_matrix.values @ pesos
    risco = np.sqrt(var)

    # Índice de Sharpe: retorno excedente por unidade de risco
    sharpe = (ret - SELIC_ANUAL) / risco

    port_retornos[i] = ret
    port_riscos[i]   = risco
    port_sharpes[i]  = sharpe
    port_pesos[i]    = pesos

print(f"✓ {N_SIM:,} portfólios simulados com sucesso!")


# ─────────────────────────────────────────────────────────────
# PASSO 5: PORTFÓLIOS ESPECIAIS
# ─────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("  PASSO 5: Identificando portfólios especiais")
print("="*60)

# Portfólio de Máximo Sharpe (melhor risco-retorno)
idx_max_sharpe = np.argmax(port_sharpes)
max_sharpe_ret   = port_retornos[idx_max_sharpe]
max_sharpe_risco = port_riscos[idx_max_sharpe]
max_sharpe_pesos = port_pesos[idx_max_sharpe]

# Portfólio de Mínima Variância
idx_min_var = np.argmin(port_riscos)
min_var_ret   = port_retornos[idx_min_var]
min_var_risco = port_riscos[idx_min_var]
min_var_pesos = port_pesos[idx_min_var]

def imprimir_portfolio(nome, pesos, retorno, risco, sharpe=None):
    print(f"\n{'─'*40}")
    print(f"  {nome}")
    print(f"{'─'*40}")
    for ticker, peso in zip(TICKERS.keys(), pesos):
        barra = "█" * int(peso * 30)
        print(f"  {ticker:6s}: {peso*100:5.1f}% {barra}")
    print(f"\n  Retorno esperado:  {retorno*100:+.2f}% a.a.")
    print(f"  Risco (Volatil.):  {risco*100:.2f}% a.a.")
    if sharpe:
        print(f"  Índice de Sharpe:  {sharpe:.3f}")

imprimir_portfolio(
    "★ PORTFÓLIO DE MÁXIMO SHARPE",
    max_sharpe_pesos, max_sharpe_ret, max_sharpe_risco,
    sharpe=port_sharpes[idx_max_sharpe]
)

imprimir_portfolio(
    "◆ PORTFÓLIO DE MÍNIMA VARIÂNCIA",
    min_var_pesos, min_var_ret, min_var_risco,
    sharpe=port_sharpes[idx_min_var]
)


# ─────────────────────────────────────────────────────────────
# PASSO 6: FRONTEIRA EFICIENTE (GRÁFICO)
# ─────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("  PASSO 6: Gerando gráfico da Fronteira Eficiente")
print("="*60)

fig, axes = plt.subplots(1, 2, figsize=(16, 7))
fig.suptitle("Análise de Portfólio de Markowitz — B3",
             fontsize=16, fontweight="bold", y=1.02)

# ── Gráfico 1: Fronteira Eficiente ──────────────────────────
ax1 = axes[0]

sc = ax1.scatter(
    port_riscos * 100,
    port_retornos * 100,
    c=port_sharpes,
    cmap="plasma",
    alpha=0.4,
    s=8,
    linewidths=0,
)
plt.colorbar(sc, ax=ax1, label="Índice de Sharpe")

# Plotar portfólios especiais
ax1.scatter(max_sharpe_risco * 100, max_sharpe_ret * 100,
            color="gold", s=200, zorder=5, marker="*",
            edgecolors="black", linewidths=0.5,
            label=f"Máx. Sharpe ({port_sharpes[idx_max_sharpe]:.2f})")

ax1.scatter(min_var_risco * 100, min_var_ret * 100,
            color="cyan", s=150, zorder=5, marker="D",
            edgecolors="black", linewidths=0.5,
            label="Mínima Variância")

# Plotar ações individuais — cores geradas dinamicamente para suportar qualquer número de ativos
cores_ativos = [cm.tab10(j) for j in range(N_ATIVOS)]  # ✅ CORRIGIDO: gera 1 cor por ativo automaticamente
for j, (ticker, nome) in enumerate(TICKERS.items()):
    ret_indiv = retorno_medio.iloc[j] * 100
    risk_indiv = np.sqrt(cov_matrix.iloc[j, j]) * 100
    ax1.scatter(risk_indiv, ret_indiv, color=cores_ativos[j],
                s=100, zorder=6, marker="o",
                edgecolors="black", linewidths=0.8)
    ax1.annotate(ticker, (risk_indiv, ret_indiv),
                 textcoords="offset points", xytext=(8, 4),
                 fontsize=9, fontweight="bold", color=cores_ativos[j])

ax1.set_xlabel("Risco (Volatilidade Anual %)", fontsize=12)
ax1.set_ylabel("Retorno Esperado Anual (%)", fontsize=12)
ax1.set_title("Fronteira Eficiente de Markowitz\n(cada ponto = 1 portfólio)", fontsize=11)
ax1.legend(fontsize=9)
ax1.grid(alpha=0.3)

# ── Gráfico 2: Alocação dos portfólios especiais ────────────
ax2 = axes[1]

nomes = list(TICKERS.keys())
x = np.arange(len(nomes))
largura = 0.35

bars1 = ax2.bar(x - largura/2, max_sharpe_pesos * 100,
                largura, label="Máx. Sharpe", color="gold",
                edgecolor="black", linewidth=0.8)
bars2 = ax2.bar(x + largura/2, min_var_pesos * 100,
                largura, label="Mín. Variância", color="cyan",
                edgecolor="black", linewidth=0.8)

# Adicionar valores nas barras
for bar in bars1:
    h = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2, h + 0.5,
             f"{h:.1f}%", ha="center", va="bottom", fontsize=9)

for bar in bars2:
    h = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2, h + 0.5,
             f"{h:.1f}%", ha="center", va="bottom", fontsize=9)

ax2.set_xlabel("Ações", fontsize=12)
ax2.set_ylabel("Peso no Portfólio (%)", fontsize=12)
ax2.set_title("Composição dos Portfólios Ótimos", fontsize=11)
ax2.set_xticks(x)
ax2.set_xticklabels(nomes)
ax2.legend()
ax2.grid(axis="y", alpha=0.3)
ax2.set_ylim(0, 100)

plt.tight_layout()
plt.savefig("markowitz_fronteira_eficiente.png", dpi=150, bbox_inches="tight")
plt.show()
print("✓ Gráfico salvo como 'markowitz_fronteira_eficiente.png'")


# ─────────────────────────────────────────────────────────────
# PASSO 7: RESUMO FINAL
# ─────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("  RESUMO: O QUE APRENDEMOS COM MARKOWITZ?")
print("="*60)
print("""
  1. DIVERSIFICAÇÃO FUNCIONA:
     O portfólio ótimo tem MENOS risco que investir em um único
     ativo, mesmo que o retorno seja similar ou maior.

  2. CORRELAÇÃO É A CHAVE:
     Misturar ativos de setores diferentes (shoppings + energia +
     farmácias) reduz mais o risco do que diversificar dentro
     do mesmo setor.

  3. DOIS PORTFÓLIOS IMPORTANTES:
     ★ Máximo Sharpe: melhor retorno por unidade de risco
     ◆ Mínima Variância: menor oscilação possível

  4. LIMITAÇÕES DO MODELO:
     - Assume distribuição normal dos retornos (nem sempre válido)
     - Baseado em dados históricos (passado ≠ futuro)
     - Não considera custos de transação ou impostos
     - Rebalancear o portfólio tem custo

  ⚠ Isso é material educacional, não recomendação de investimento.
""")