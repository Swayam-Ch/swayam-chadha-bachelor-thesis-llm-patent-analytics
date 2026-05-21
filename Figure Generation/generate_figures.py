"""
generate_figures.py
Reproduces all figures for:
  "LLM-Assisted Patent Analytics for Mapping AI Innovation Trends"
  Swayam Chadha, LMU Munich, 2026

Input:  /mnt/user-data/uploads/classifications_final.csv
Output: figures/fig1_growth.pdf/.png
        figures/fig2_technique.pdf/.png
        figures/fig3_domains.pdf/.png
        figures/fig4_orientation.pdf/.png
        figures/fig5_heatmap.pdf/.png

Usage:
    python generate_figures.py
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.stats import linregress

# ── Output directory ───────────────────────────────────────────────────────
os.makedirs('figures', exist_ok=True)

# ── Load data ──────────────────────────────────────────────────────────────
df = pd.read_csv('classifications_final.csv', low_memory=False)
genuine = df[df['is_genuine_ai'] == True].copy()
total = len(genuine)

# ── Technique remapping (method-only taxonomy) ─────────────────────────────
TECH_MAP = {
    'neural_network_general': 'neural_network_general',
    'computer_vision_cnn':    'cnn_convolutional',
    'transformer_llm':        'transformer_attention',
    'classical_ml':           'classical_ml',
    'generative_model':       'generative_model',
    'speech_audio':           'other',
    'reinforcement_learning': 'reinforcement_learning',
    'optimization':           'optimization',
}
genuine['tech_grouped'] = genuine['ai_technique'].apply(
    lambda t: TECH_MAP.get(t, 'other'))

TECH_COLORS = {
    'neural_network_general': '#1f4e79',
    'cnn_convolutional':      '#e07b39',
    'transformer_attention':  '#2e8b57',
    'classical_ml':           '#9b59b6',
    'generative_model':       '#c0392b',
    'reinforcement_learning': '#17a589',
    'optimization':           '#b7b7a4',
    'other':                  '#e0e0e0',
}
TECH_LABELS = {
    'neural_network_general': 'Neural network (general)',
    'cnn_convolutional':      'CNN / convolutional',
    'transformer_attention':  'Transformer / attention',
    'classical_ml':           'Classical ML',
    'generative_model':       'Generative model',
    'reinforcement_learning': 'Reinforcement learning',
    'optimization':           'Optimization',
    'other':                  'Other',
}

STACK_ORDER = ['neural_network_general', 'cnn_convolutional', 'transformer_attention',
               'classical_ml', 'generative_model', 'reinforcement_learning',
               'optimization', 'other']

# ══════════════════════════════════════════════════════════════════════════
# FIG 1 — Growth of genuine AI patents with exponential fit + discontinuity
# ══════════════════════════════════════════════════════════════════════════
print("Generating fig1_growth...")

by_year = genuine.groupby('year').size().sort_index()
years   = np.array(by_year.index.tolist())
counts  = np.array(by_year.values.tolist())
x0      = 2010

def exp_func(x, a, b):
    return a * np.exp(b * (x - x0))

popt, _   = curve_fit(exp_func, years, counts, p0=[3000, 0.2])
r_all     = popt[1]
growth_all = (np.exp(r_all) - 1) * 100
r2_all    = 1 - np.sum((counts - exp_func(years, *popt))**2) / \
              np.sum((counts - counts.mean())**2)

x_smooth = np.linspace(2010, 2023, 300)

count_2019 = int(by_year[2019])
count_2020 = int(by_year[2020])
jump       = count_2020 - count_2019
pct_jump   = jump / count_2019 * 100
pre_avg    = np.mean(np.diff(counts[:10]))
post_avg   = np.mean(np.diff(counts[9:]))

fig, ax = plt.subplots(figsize=(10, 5.5))
ax.axvspan(2019.5, 2023.5, color='lightgrey', alpha=0.45, zorder=0)
ax.bar(years, counts, color='steelblue', width=0.7, zorder=2,
       label='Annual patent count')
ax.plot(x_smooth, exp_func(x_smooth, *popt), color='#c0392b', linewidth=2,
        zorder=3,
        label=f'Exponential fit: r = {r_all:.3f} yr$^{{-1}}$ '
              f'({growth_all:.1f}%/yr, $R^2$ = {r2_all:.3f})')

ax.annotate('', xy=(2020, count_2020 + 500), xytext=(2020, count_2019 + 500),
            arrowprops=dict(arrowstyle='<->', color='#2c3e50', lw=1.5))
ax.text(2020.35, (count_2019 + count_2020) / 2,
        f'+{jump/1000:.1f}k\n({pct_jump:.0f}%) in\none year',
        fontsize=8.5, va='center', color='#2c3e50')
ax.text(2014.5, 38000,
        f'Avg. +{pre_avg:.0f} patents/yr\n(2010–2019)',
        fontsize=8.5, ha='center', color='#2980b9', fontweight='bold')
ax.text(2021.5, 38000,
        f'Avg. +{post_avg:.0f} patents/yr\n(2020–2023)',
        fontsize=8.5, ha='center', color='#c0392b', fontweight='bold')

ax.set_xlabel('Publication year', fontsize=12)
ax.set_ylabel('Genuine AI patents', fontsize=12)
ax.set_xticks(years)
ax.set_xticklabels([str(y) if y % 2 == 0 else '' for y in years])
ax.yaxis.set_major_formatter(
    plt.FuncFormatter(lambda x, _: f'{int(x/1000)}k' if x >= 1000 else str(int(x))))
ax.legend(loc='upper left', fontsize=9, frameon=False)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.set_xlim(2009.3, 2023.7)
ax.set_ylim(0, 43000)
plt.tight_layout()
plt.savefig('figures/fig1_growth.pdf', bbox_inches='tight')
plt.savefig('figures/fig1_growth.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Done.")

# ══════════════════════════════════════════════════════════════════════════
# FIG 2 — Technique composition: share (a) + absolute counts (b)
# ══════════════════════════════════════════════════════════════════════════
print("Generating fig2_technique...")

by_year_tech  = genuine.groupby(['year', 'tech_grouped']).size().unstack(fill_value=0)
by_year_total = by_year_tech.sum(axis=1)
shares        = by_year_tech.div(by_year_total, axis=0) * 100
stack_order   = [t for t in STACK_ORDER if t in shares.columns]
line_techs    = [t for t in stack_order if t != 'other']
years_t       = by_year_tech.index.tolist()
end_vals      = {tech: by_year_tech.loc[2023, tech] for tech in line_techs}

label_y = {
    'neural_network_general': 28000,
    'cnn_convolutional':       5500,
    'transformer_attention':   3800,
    'classical_ml':            2600,
    'generative_model':        1800,
    'reinforcement_learning':  1200,
    'optimization':             550,
}

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 7))

# Panel A: share
bottom = np.zeros(len(years_t))
for tech in stack_order:
    vals = shares[tech].values
    ax1.fill_between(years_t, bottom, bottom + vals,
                     color=TECH_COLORS[tech], alpha=0.92, label=TECH_LABELS[tech])
    bottom += vals
ax1.set_xlim(2010, 2023)
ax1.set_ylim(0, 100)
ax1.set_xlabel('Publication year', fontsize=11)
ax1.set_ylabel('Share of AI patents (%)', fontsize=11)
ax1.set_xticks(years_t)
ax1.set_title('(a) Technique composition — annual share', fontsize=11, loc='left', pad=6)
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)
ax1.tick_params(labelsize=9)

# Panel B: absolute log scale
for tech in line_techs:
    vals = by_year_tech[tech].values
    lw = 2.5 if tech == 'neural_network_general' else 1.8
    ax2.plot(years_t, vals, color=TECH_COLORS[tech], linewidth=lw,
             marker='o', markersize=3.5)
    ax2.annotate('', xy=(2023, end_vals[tech]),
                 xytext=(2023.6, label_y[tech]),
                 arrowprops=dict(arrowstyle='-', color=TECH_COLORS[tech], lw=0.8))
    ax2.text(2023.7, label_y[tech], TECH_LABELS[tech],
             fontsize=8, color=TECH_COLORS[tech], va='center')
ax2.set_yscale('log')
ax2.set_xlim(2010, 2027.5)
ax2.set_ylim(30, 60000)
ax2.set_xlabel('Publication year', fontsize=11)
ax2.set_ylabel('AI patents (log scale)', fontsize=11)
ax2.set_xticks(years_t)
ax2.set_title('(b) Technique composition — absolute counts (log scale)',
              fontsize=11, loc='left', pad=6)
ax2.yaxis.set_major_formatter(
    plt.FuncFormatter(lambda x, _: f'{int(x/1000)}k' if x >= 1000 else str(int(x))))
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)
ax2.tick_params(labelsize=9)
ax2.grid(axis='y', which='both', alpha=0.15, lw=0.5)

handles, lbls = ax1.get_legend_handles_labels()
fig.legend(handles, lbls, loc='upper right', bbox_to_anchor=(0.98, 0.97),
           fontsize=9, frameon=False, ncol=1)
plt.tight_layout(rect=[0, 0, 0.82, 1])
plt.savefig('figures/fig2_technique.pdf', bbox_inches='tight')
plt.savefig('figures/fig2_technique.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Done.")

# ══════════════════════════════════════════════════════════════════════════
# FIG 3 — Top 10 application domains
# ══════════════════════════════════════════════════════════════════════════
print("Generating fig3_domains...")

DOMAIN_LABELS = {
    'computer_vision_imaging':   'Imaging',
    'nlp_text':                  'NLP / text',
    'healthcare_medical':        'Healthcare',
    'autonomous_systems':        'Autonomous systems',
    'ai_methods_infrastructure': 'AI methods &\ninfrastructure',
    'consumer_electronics':      'Consumer electronics',
    'industrial_manufacturing':  'Industrial / mfg.',
    'speech_audio':              'Speech & audio',
    'finance_business':          'Finance & business',
    'robotics':                  'Robotics',
    'security_privacy':          'Security & privacy',
    'networking_infrastructure': 'Networking',
    'gaming_entertainment':      'Gaming & entertainment',
    'productivity_enterprise':   'Productivity / enterprise',
    'scientific_research':       'Scientific research',
    'other':                     'Other',
}

domain_counts = genuine['application_domain'].value_counts()
top10         = domain_counts.head(10)
labels_d      = [DOMAIN_LABELS.get(d, d) for d in top10.index]
pcts_d        = top10.values / total * 100
colors_d      = plt.cm.Blues(np.linspace(0.85, 0.35, len(top10)))

fig, ax = plt.subplots(figsize=(8, 5))
ax.barh(range(len(labels_d)), pcts_d, color=colors_d, height=0.6)
ax.set_yticks(range(len(labels_d)))
ax.set_yticklabels(labels_d, fontsize=9.5)
ax.set_xlabel('Share of AI patents (%)', fontsize=10)
ax.set_xlim(0, 25)
ax.invert_yaxis()
for i, p in enumerate(pcts_d):
    ax.text(p + 0.3, i, f'{p:.1f}%', va='center', fontsize=9)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.tick_params(left=False)
plt.tight_layout()
plt.savefig('figures/fig3_domains.pdf', bbox_inches='tight')
plt.savefig('figures/fig3_domains.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Done.")

# ══════════════════════════════════════════════════════════════════════════
# FIG 4 — Innovation orientation + contribution type
# ══════════════════════════════════════════════════════════════════════════
print("Generating fig4_orientation...")

by_year_ori   = genuine.groupby(['year', 'innovation_orientation']).size().unstack(fill_value=0)
by_year_ori_t = by_year_ori.sum(axis=1)
by_year_pct   = by_year_ori.div(by_year_ori_t, axis=0) * 100

contrib        = genuine['contribution_type'].value_counts()
CONTRIB_LABELS = {
    'algorithmic':                'Algorithmic',
    'application_implementation': 'Application /\nimplementation',
    'system_integration':         'System\nintegration',
    'architectural':              'Architectural',
    'data_method':                'Data method',
}
contrib_names = [CONTRIB_LABELS.get(k, k) for k in contrib.index]
contrib_pcts  = contrib.values / total * 100

years_o      = by_year_pct.index.tolist()
applied_vals = by_year_pct['applied'].values
fund_vals    = by_year_pct['fundamental'].values \
               if 'fundamental' in by_year_pct else np.zeros(len(years_o))

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 3.5))

ax1.stackplot(years_o, applied_vals, fund_vals,
              labels=['Applied', 'Fundamental'],
              colors=['#2e75b6', '#f4a261'], alpha=0.9)
ax1.annotate(f'{applied_vals[0]:.1f}%',
             xy=(2010, applied_vals[0] - 3),
             fontsize=8.5, color='white', ha='left', va='top', fontweight='bold')
ax1.annotate(f'{applied_vals[-1]:.1f}%',
             xy=(2022.8, applied_vals[-1] - 3),
             fontsize=8.5, color='white', ha='right', va='top', fontweight='bold')
ax1.annotate('', xy=(2023, applied_vals[-1]), xytext=(2010, applied_vals[0]),
             arrowprops=dict(arrowstyle='->', color='white', lw=1.5,
                             connectionstyle='arc3,rad=0'))
ax1.set_xlim(2010, 2023)
ax1.set_ylim(0, 100)
ax1.set_xlabel('Publication year', fontsize=9)
ax1.set_ylabel('Share of AI patents (%)', fontsize=9)
ax1.set_title('(a) Innovation orientation over time', fontsize=9, pad=4)
ax1.legend(loc='lower left', fontsize=8, frameon=False)
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)
ax1.tick_params(labelsize=8)

colors_b = plt.cm.Blues(np.linspace(0.85, 0.35, len(contrib_pcts)))
ax2.barh(range(len(contrib_names)), contrib_pcts, color=colors_b, height=0.5)
ax2.set_yticks(range(len(contrib_names)))
ax2.set_yticklabels(contrib_names, fontsize=8)
ax2.set_xlabel('Share of AI patents (%)', fontsize=9)
ax2.set_title('(b) Contribution type', fontsize=9, pad=4)
ax2.set_xlim(0, 48)
for i, p in enumerate(contrib_pcts):
    ax2.text(p + 0.5, i, f'{p:.1f}%', va='center', fontsize=8)
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)
ax2.spines['left'].set_visible(False)
ax2.tick_params(left=False, labelsize=8)
plt.tight_layout(pad=0.8)
plt.savefig('figures/fig4_orientation.pdf', bbox_inches='tight')
plt.savefig('figures/fig4_orientation.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Done.")

# ══════════════════════════════════════════════════════════════════════════
# FIG 5 — Technique × domain heatmap
# ══════════════════════════════════════════════════════════════════════════
print("Generating fig5_heatmap...")

genuine['tech_grouped2'] = genuine['ai_technique'].apply(
    lambda t: 'speech_audio_tech' if t == 'speech_audio' else TECH_MAP.get(t, 'other'))

TOP_TECHS = ['neural_network_general', 'cnn_convolutional', 'transformer_attention',
             'classical_ml', 'generative_model', 'speech_audio_tech',
             'reinforcement_learning', 'optimization']
TOP_DOMAINS = ['computer_vision_imaging', 'nlp_text', 'healthcare_medical',
               'autonomous_systems', 'ai_methods_infrastructure',
               'consumer_electronics', 'industrial_manufacturing', 'speech_audio']

HEATMAP_TECH_LABELS = {
    'neural_network_general': 'Neural network\n(general)',
    'cnn_convolutional':      'CNN /\nconvolutional',
    'transformer_attention':  'Transformer /\nattention',
    'classical_ml':           'Classical ML',
    'generative_model':       'Generative\nmodel',
    'speech_audio_tech':      'Speech &\naudio',
    'reinforcement_learning': 'Reinforcement\nlearning',
    'optimization':           'Optimization',
}
HEATMAP_DOMAIN_LABELS = {
    'computer_vision_imaging':   'Computer\nvision',
    'nlp_text':                  'NLP /\ntext',
    'healthcare_medical':        'Healthcare',
    'autonomous_systems':        'Autonomous\nsystems',
    'ai_methods_infrastructure': 'AI methods &\ninfrastructure',
    'consumer_electronics':      'Consumer\nelectronics',
    'industrial_manufacturing':  'Industrial\n/ mfg.',
    'speech_audio':              'Speech\n& audio',
}

sub  = genuine[genuine['tech_grouped2'].isin(TOP_TECHS) &
               genuine['application_domain'].isin(TOP_DOMAINS)]
heat = sub.groupby(['tech_grouped2', 'application_domain']).size().unstack(fill_value=0)
heat = heat.reindex(index=TOP_TECHS, columns=TOP_DOMAINS, fill_value=0)
heat_norm = heat.div(heat.sum(axis=1), axis=0) * 100

fig, ax = plt.subplots(figsize=(10, 5))
im = ax.imshow(heat_norm.values, cmap=plt.cm.Blues, aspect='auto', vmin=0, vmax=75)
for i in range(len(TOP_TECHS)):
    for j in range(len(TOP_DOMAINS)):
        val = heat_norm.values[i, j]
        ax.text(j, i, f'{val:.1f}%', ha='center', va='center',
                fontsize=8, color='white' if val > 40 else '#333333')
ax.set_xticks(range(len(TOP_DOMAINS)))
ax.set_xticklabels([HEATMAP_DOMAIN_LABELS[d] for d in TOP_DOMAINS], fontsize=8.5)
ax.set_yticks(range(len(TOP_TECHS)))
ax.set_yticklabels([HEATMAP_TECH_LABELS[t] for t in TOP_TECHS], fontsize=8.5)
ax.set_xlabel('Application domain', fontsize=10)
ax.set_ylabel('')
ax.tick_params(length=0)
cbar = fig.colorbar(im, ax=ax, shrink=0.85, pad=0.02)
cbar.set_label('Share of application domain (%)', fontsize=8.5)
cbar.ax.tick_params(labelsize=8)
fig.tight_layout(pad=1.2)
plt.savefig('figures/fig5_heatmap.pdf', bbox_inches='tight')
plt.savefig('figures/fig5_heatmap.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Done.")

print("\nAll figures saved to figures/")
