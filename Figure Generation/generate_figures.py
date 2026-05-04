"""
Figure generation script for:
LLM-Assisted Patent Analytics for Mapping AI Innovation Trends
Bachelor's Thesis — LMU Munich
Author: Swayam Chadha

Requires: classifications_final.csv in the same directory
Output:   fig1_growth.pdf/png
          fig2_technique.pdf/png
          fig3_domains.pdf/png
          fig4_orientation.pdf/png
          fig5_heatmap.pdf/png
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# ── Load data ────────────────────────────────────────────────
df = pd.read_csv('classifications_final.csv', low_memory=False)
genuine = df[df['is_genuine_ai'] == True].copy()
total = len(genuine)

# ============================================================
# Figure 1 — Genuine AI patents by year (bar chart + growth curve)
# ============================================================

by_year = genuine.groupby('year').size().sort_index()

fig, ax = plt.subplots(figsize=(10, 5.5))
years = by_year.index.tolist()
counts = by_year.values.tolist()

ax.axvspan(2019.5, 2023.5, color='lightgrey', alpha=0.4, zorder=0)
ax.bar(years, counts, color='steelblue', width=0.7, zorder=2)

x_smooth = np.linspace(2010, 2023, 300)
log_counts = np.log(counts)
coeffs = np.polyfit(years, log_counts, 1)
y_smooth = np.exp(np.polyval(coeffs, x_smooth))
ax.plot(x_smooth, y_smooth, color='#c0392b', linewidth=2, zorder=3)

ax.annotate(
    '×11.8 growth\n2010→2023',
    xy=(2016.5, np.exp(np.polyval(coeffs, 2016.5))),
    xytext=(2015.2, 32000),
    color='#c0392b',
    fontsize=10,
    fontstyle='italic',
    arrowprops=dict(arrowstyle='->', color='#c0392b', lw=1.2),
    ha='center'
)

ax.set_xlabel('Publication year', fontsize=12)
ax.set_ylabel('Genuine AI patents', fontsize=12)
ax.set_xticks(years)
ax.set_xticklabels([str(y) if y % 2 == 0 else '' for y in years])
ax.yaxis.set_major_formatter(plt.FuncFormatter(
    lambda x, _: f'{int(x/1000)}k' if x >= 1000 else str(int(x))))
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.set_xlim(2009.3, 2023.7)
ax.set_ylim(0, 43000)

plt.tight_layout()
plt.savefig('fig1_growth.pdf', bbox_inches='tight')
plt.savefig('fig1_growth.png', dpi=150, bbox_inches='tight')
plt.close()
print("Figure 1 done")

# ============================================================
# Figure 2 — AI technique composition stacked area chart
# ============================================================

main_techs = ['neural_network_general', 'computer_vision_cnn', 'transformer_llm',
              'classical_ml', 'generative_model', 'speech_audio',
              'reinforcement_learning', 'optimization']

genuine['tech_grouped'] = genuine['ai_technique'].apply(
    lambda t: t if t in main_techs else 'other')

by_year_tech = genuine.groupby(['year', 'tech_grouped']).size().unstack(fill_value=0)
by_year_total = by_year_tech.sum(axis=1)
shares = by_year_tech.div(by_year_total, axis=0) * 100

stack_order = ['neural_network_general', 'computer_vision_cnn', 'transformer_llm',
               'classical_ml', 'generative_model', 'speech_audio',
               'reinforcement_learning', 'optimization', 'other']
stack_order = [t for t in stack_order if t in shares.columns]

colors = {
    'neural_network_general': '#1f4e79',
    'computer_vision_cnn':    '#2e75b6',
    'transformer_llm':        '#5ba3d9',
    'classical_ml':           '#f4a261',
    'generative_model':       '#e76f51',
    'speech_audio':           '#8ecae6',
    'reinforcement_learning': '#95d5b2',
    'optimization':           '#b7b7a4',
    'other':                  '#e0e0e0',
}

labels = {
    'neural_network_general': 'Neural network (general)',
    'computer_vision_cnn':    'Computer vision / CNN',
    'transformer_llm':        'Transformer / LLM',
    'classical_ml':           'Classical ML',
    'generative_model':       'Generative model',
    'speech_audio':           'Speech & audio',
    'reinforcement_learning': 'Reinforcement learning',
    'optimization':           'Optimization',
    'other':                  'Other',
}

fig, ax = plt.subplots(figsize=(11, 5.5))
years = shares.index.tolist()
bottom = np.zeros(len(years))

for tech in stack_order:
    vals = shares[tech].values
    ax.fill_between(years, bottom, bottom + vals,
                    color=colors[tech], alpha=0.92, label=labels[tech])
    bottom += vals

ax.set_xlim(2010, 2023)
ax.set_ylim(0, 100)
ax.set_xlabel('Publication year', fontsize=12)
ax.set_ylabel('Share of genuine AI patents (%)', fontsize=12)
ax.set_xticks(years)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
handles, lbls = ax.get_legend_handles_labels()
ax.legend(handles, lbls, loc='center left', bbox_to_anchor=(1.01, 0.5),
          fontsize=9, frameon=False)

plt.tight_layout()
plt.savefig('fig2_technique.pdf', bbox_inches='tight')
plt.savefig('fig2_technique.png', dpi=150, bbox_inches='tight')
plt.close()
print("Figure 2 done")

# ============================================================
# Figure 3 — Top 10 application domains horizontal bar chart
# ============================================================

domains = genuine['application_domain'].value_counts().head(10)

domain_labels = {
    'computer_vision_imaging':   'Computer vision / imaging',
    'nlp_text':                  'NLP / text',
    'healthcare_medical':        'Healthcare & medical',
    'autonomous_systems':        'Autonomous systems',
    'ai_methods_infrastructure': 'AI methods & infrastructure',
    'consumer_electronics':      'Consumer electronics',
    'industrial_manufacturing':  'Industrial manufacturing',
    'speech_audio':              'Speech & audio',
    'security_privacy':          'Security & privacy',
    'finance_business':          'Finance & business',
}

names = [domain_labels[d] for d in domains.index]
pcts  = (domains.values / total * 100).tolist()
names_r = names[::-1]
pcts_r  = pcts[::-1]

fig, ax = plt.subplots(figsize=(9, 5.5))
bar_colors = plt.cm.Blues(np.linspace(0.35, 0.85, len(names_r)))
ax.barh(range(len(names_r)), pcts_r, color=bar_colors, height=0.6)
ax.set_yticks(range(len(names_r)))
ax.set_yticklabels(names_r, fontsize=10)
ax.set_xlabel('Share of genuine AI patents (%)', fontsize=11)
ax.set_xlim(0, 23)
for i, p in enumerate(pcts_r):
    ax.text(p + 0.3, i, f'{p:.1f}%', va='center', fontsize=9.5)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.tick_params(left=False)

plt.tight_layout()
plt.savefig('fig3_domains.pdf', bbox_inches='tight')
plt.savefig('fig3_domains.png', dpi=150, bbox_inches='tight')
plt.close()
print("Figure 3 done")

# ============================================================
# Figure 4 — Innovation orientation + contribution type (2-panel)
# ============================================================

by_year_orient = genuine.groupby(
    ['year', 'innovation_orientation']).size().unstack(fill_value=0)
by_year_orient_total = by_year_orient.sum(axis=1)
by_year_orient_pct = by_year_orient.div(by_year_orient_total, axis=0) * 100

contrib = genuine['contribution_type'].value_counts()
contrib_label_map = {
    'algorithmic':                'Algorithmic',
    'application_implementation': 'Application /\nimplementation',
    'system_integration':         'System\nintegration',
    'architectural':              'Architectural',
    'data_method':                'Data method',
}
contrib_names = [contrib_label_map.get(k, k) for k in contrib.index]
contrib_pcts  = contrib.values / total * 100

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 3.8))

years = by_year_orient_pct.index.tolist()
applied_vals = by_year_orient_pct['applied'].values \
    if 'applied' in by_year_orient_pct else np.zeros(len(years))
fund_vals = by_year_orient_pct['fundamental'].values \
    if 'fundamental' in by_year_orient_pct else np.zeros(len(years))

ax1.stackplot(years, applied_vals, fund_vals,
              labels=['Applied', 'Fundamental'],
              colors=['#2e75b6', '#f4a261'], alpha=0.9)
ax1.set_xlim(2010, 2023)
ax1.set_ylim(0, 100)
ax1.set_xlabel('Publication year', fontsize=10)
ax1.set_ylabel('Share of genuine AI patents (%)', fontsize=10)
ax1.set_title('(a) Innovation orientation over time', fontsize=10, pad=6)
ax1.legend(loc='lower left', fontsize=8.5, frameon=False)
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)

colors_b = plt.cm.Blues(np.linspace(0.85, 0.35, len(contrib_pcts)))
ax2.barh(range(len(contrib_names)), contrib_pcts, color=colors_b, height=0.55)
ax2.set_yticks(range(len(contrib_names)))
ax2.set_yticklabels(contrib_names, fontsize=9.5)
ax2.set_xlabel('Share of genuine AI patents (%)', fontsize=10)
ax2.set_title('(b) Contribution type', fontsize=10, pad=6)
ax2.set_xlim(0, 48)
for i, p in enumerate(contrib_pcts):
    ax2.text(p + 0.5, i, f'{p:.1f}%', va='center', fontsize=9)
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)
ax2.spines['left'].set_visible(False)
ax2.tick_params(left=False)

plt.tight_layout()
plt.savefig('fig4_orientation.pdf', bbox_inches='tight')
plt.savefig('fig4_orientation.png', dpi=150, bbox_inches='tight')
plt.close()
print("Figure 4 done")

# ============================================================
# Figure 5 — Technique x domain specialisation heatmap
# ============================================================

top_techs   = genuine['ai_technique'].value_counts().head(8).index.tolist()
top_domains = genuine['application_domain'].value_counts().head(8).index.tolist()

subset = genuine[genuine['ai_technique'].isin(top_techs) &
                 genuine['application_domain'].isin(top_domains)]

ct = pd.crosstab(subset['ai_technique'], subset['application_domain'])
ct = ct.reindex(index=top_techs, columns=top_domains, fill_value=0)
ct_pct = ct.div(ct.sum(axis=1), axis=0) * 100

tech_label_map = {
    'neural_network_general': 'Neural network (general)',
    'computer_vision_cnn':    'Computer vision / CNN',
    'transformer_llm':        'Transformer / LLM',
    'classical_ml':           'Classical ML',
    'generative_model':       'Generative model',
    'speech_audio':           'Speech & audio',
    'reinforcement_learning': 'Reinforcement learning',
    'optimization':           'Optimization',
}
domain_label_map = {
    'computer_vision_imaging':   'Computer\nvision',
    'nlp_text':                  'NLP /\ntext',
    'healthcare_medical':        'Healthcare',
    'autonomous_systems':        'Autonomous\nsystems',
    'ai_methods_infrastructure': 'AI infra-\nstructure',
    'consumer_electronics':      'Consumer\nelectronics',
    'industrial_manufacturing':  'Industrial\n/ mfg.',
    'speech_audio':              'Speech\n& audio',
}

ct_pct.index   = [tech_label_map.get(t, t) for t in ct_pct.index]
ct_pct.columns = [domain_label_map.get(d, d) for d in ct_pct.columns]

fig, ax = plt.subplots(figsize=(12, 6))
im = ax.imshow(ct_pct.values, cmap='Blues', aspect='auto',
               vmin=0, vmax=ct_pct.values.max())

ax.set_xticks(range(len(ct_pct.columns)))
ax.set_xticklabels(ct_pct.columns, rotation=0, ha='center', fontsize=9.5)
ax.set_yticks(range(len(ct_pct.index)))
ax.set_yticklabels(ct_pct.index, fontsize=10)

for i in range(len(ct_pct.index)):
    for j in range(len(ct_pct.columns)):
        val = ct_pct.values[i, j]
        color = 'white' if val > 40 else 'black'
        ax.text(j, i, f'{val:.1f}%', ha='center', va='center',
                fontsize=8.5, color=color)

cbar = plt.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
cbar.set_label('Row share (%)', fontsize=10)
ax.set_xlabel('Application domain', fontsize=11)
ax.set_ylabel('AI technique', fontsize=11)

plt.tight_layout()
plt.savefig('fig5_heatmap.pdf', bbox_inches='tight')
plt.savefig('fig5_heatmap.png', dpi=150, bbox_inches='tight')
plt.close()
print("Figure 5 done")

print("\nAll figures generated successfully.")
