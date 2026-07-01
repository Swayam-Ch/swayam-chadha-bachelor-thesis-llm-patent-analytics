"""
generate_figures.py
Reproduces all figures for:
  "LLM-Assisted Patent Analytics for Mapping AI Innovation Trends"
  Swayam Chadha, LMU Munich, 2026

Input:  classifications_final.csv
        epo_results_qwen_bigquery.csv  (for fig6)
Output: figures/fig1_growth.pdf/.png
        figures/fig8_regression.pdf/.png
        figures/fig9_country_breakdown.pdf/.png
        figures/fig2_technique.pdf/.png
        figures/fig3_domains.pdf/.png
        figures/fig4_orientation.pdf/.png
        figures/fig5_heatmap.pdf/.png
        figures/fig6_epo_comparison.pdf/.png

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

# ── Load USPTO data ────────────────────────────────────────────────────────
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

popt, _    = curve_fit(exp_func, years, counts, p0=[3000, 0.2])
r_all      = popt[1]
growth_all = (np.exp(r_all) - 1) * 100
r2_all     = 1 - np.sum((counts - exp_func(years, *popt))**2) / \
               np.sum((counts - counts.mean())**2)

x_smooth   = np.linspace(2010, 2023, 300)
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

sub       = genuine[genuine['tech_grouped2'].isin(TOP_TECHS) &
                    genuine['application_domain'].isin(TOP_DOMAINS)]
heat      = sub.groupby(['tech_grouped2', 'application_domain']).size().unstack(fill_value=0)
heat      = heat.reindex(index=TOP_TECHS, columns=TOP_DOMAINS, fill_value=0)
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

# ══════════════════════════════════════════════════════════════════════════
# FIG 6 — USPTO vs EPO comparison
# ══════════════════════════════════════════════════════════════════════════
print("Generating fig6_epo_comparison...")

import os as _os
if not _os.path.exists('epo_results_qwen_bigquery.csv'):
    print("  epo_results_qwen_bigquery.csv not found — skipping fig6")
else:
    epo = pd.read_csv('epo_results_qwen_bigquery.csv')
    epo['is_genuine_ai'] = epo['is_genuine_ai'].map(
        {'True': True, 'False': False, True: True, False: False})
    epo_genuine = epo[epo['is_genuine_ai'] == True].copy()

    TECH_MAP_COMPARISON = {
        'neural_network_general': 'Neural net\n(general)',
        'computer_vision_cnn':    'CNN /\nconvolutional',
        'transformer_llm':        'Transformer /\nattention',
        'classical_ml':           'Classical ML',
        'generative_model':       'Generative\nmodel',
        'speech_audio':           'Speech\n& audio',
    }
    genuine['tech_cmp']     = genuine['ai_technique'].map(TECH_MAP_COMPARISON)
    epo_genuine['tech_cmp'] = epo_genuine['ai_technique'].map(TECH_MAP_COMPARISON)

    years_all = list(range(2010, 2024))

    uspto_by_year = genuine.groupby('year').size().reindex(years_all, fill_value=0)
    epo_by_year   = epo_genuine.groupby('year').size().reindex(years_all, fill_value=0)
    def applied_share(df, years):
        result = []
        for y in years:
            sub = df[df['year'] == y]
            if len(sub) == 0:
                result.append(np.nan)
                continue
            result.append((sub['innovation_orientation'] == 'applied').sum() / len(sub) * 100)
        return result

    uspto_applied = applied_share(genuine, years_all)
    epo_applied   = applied_share(epo_genuine, years_all)
    slope_u, int_u, *_ = linregress(years_all, uspto_applied)
    slope_e, int_e, *_ = linregress(years_all, epo_applied)
    trend_u = [int_u + slope_u * y for y in years_all]
    trend_e = [int_e + slope_e * y for y in years_all]

    techs_cmp = ['Neural net\n(general)', 'CNN /\nconvolutional', 'Classical ML',
                 'Transformer /\nattention', 'Speech\n& audio', 'Generative\nmodel']

    def tech_shares(df, techs):
        t = len(df)
        return [df[df['tech_cmp'] == tc].shape[0] / t * 100 for tc in techs]

    uspto_tech = tech_shares(genuine, techs_cmp)
    epo_tech   = tech_shares(epo_genuine, techs_cmp)

    USPTO_COLOR = '#1f4e79'
    EPO_COLOR   = '#c0392b'

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    ax = axes[0]
    ax.plot(years_all, uspto_by_year.values, color=USPTO_COLOR, lw=2.2,
            marker='o', markersize=4, label=f'USPTO (n={len(genuine):,})')
    ax.plot(years_all, epo_by_year.values,   color=EPO_COLOR,   lw=2.2,
            marker='s', markersize=4, label=f'EPO (n={len(epo_genuine):,})')
    ax.set_yscale('log')
    ax.set_xlabel('Publication year', fontsize=10)
    ax.set_ylabel('Genuine AI patents (log scale)', fontsize=10)
    ax.set_title('(a) Annual genuine AI patent counts', fontsize=10, loc='left')
    ax.legend(fontsize=9, frameon=False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_xlim(2010, 2023)
    ax.tick_params(labelsize=8)
    ax.set_xticks(years_all[::2])
    ax.yaxis.set_major_formatter(plt.FuncFormatter(
        lambda x, _: f"{int(x/1000)}k" if x >= 1000 else str(int(x))))
    ax.grid(axis='y', which='both', alpha=0.12, lw=0.5)

    ax = axes[1]
    ax.plot(years_all, uspto_applied, color=USPTO_COLOR, lw=2.2,
            marker='o', markersize=4, label='USPTO')
    ax.plot(years_all, epo_applied,   color=EPO_COLOR,   lw=2.2,
            marker='s', markersize=4, label='EPO')
    ax.plot(years_all, trend_u, color=USPTO_COLOR, lw=1.2, linestyle='--', alpha=0.7)
    ax.plot(years_all, trend_e, color=EPO_COLOR,   lw=1.2, linestyle='--', alpha=0.7)
    ax.set_xlabel('Publication year', fontsize=10)
    ax.set_ylabel('Applied patents (%)', fontsize=10)
    ax.set_title('(b) Innovation orientation — applied share', fontsize=10, loc='left')
    ax.legend(fontsize=9, frameon=False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_xlim(2010, 2023)
    ax.set_ylim(50, 100)
    ax.tick_params(labelsize=8)
    ax.set_xticks(years_all[::2])

    ax = axes[2]
    x      = np.arange(len(techs_cmp))
    width  = 0.35
    bars_u = ax.barh(x + width/2, uspto_tech, width,
                     color=USPTO_COLOR, alpha=0.85, label='USPTO')
    bars_e = ax.barh(x - width/2, epo_tech,   width,
                     color=EPO_COLOR,   alpha=0.85, label='EPO')
    ax.set_yticks(x)
    ax.set_yticklabels(techs_cmp, fontsize=8.5)
    ax.set_xlabel('Share of genuine AI patents (%)', fontsize=10)
    ax.set_title('(c) Technique composition — USPTO vs EPO', fontsize=10, loc='left')
    ax.legend(fontsize=9, frameon=False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.tick_params(left=False, labelsize=8)
    for b, v in zip(bars_u, uspto_tech):
        ax.text(v + 0.3, b.get_y() + b.get_height()/2, f'{v:.1f}%',
                va='center', fontsize=7.5, color=USPTO_COLOR)
    for b, v in zip(bars_e, epo_tech):
        ax.text(v + 0.3, b.get_y() + b.get_height()/2, f'{v:.1f}%',
                va='center', fontsize=7.5, color=EPO_COLOR)

    plt.tight_layout(pad=1.5)
    plt.savefig('figures/fig6_epo_comparison.pdf', bbox_inches='tight')
    plt.savefig('figures/fig6_epo_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  Done.")


# ══════════════════════════════════════════════════════════════════════════
# FIG 7 — Emerging AI techniques (keyword search in reasoning field)
# ══════════════════════════════════════════════════════════════════════════
print("Generating fig7_emerging_tech...")

EMERGING = {
    'Foundation models / LLMs': (
        r'language.model|large.language|llm\b|foundation.model|gpt\b|bert\b|'
        r'pre.?trained.{0,20}model|generative.pre.?train|instruction.tun|'
        r'chat.model|transformer.model'
    ),
    'Federated learning': (
        r'federated.learn|federated.train|federated.optim|'
        r'federat\w+.aggregat|split.learning'
    ),
    'Agents / agentic AI': (
        r'\bagent\b|agentic|multi.?agent|autonomous.agent|'
        r'tool.?use|tool.?call|planning.agent|llm.agent'
    ),
    'Model distillation': (
        r'distill|knowledge.transfer.{0,20}model|'
        r'teacher.student|student.model|model.compression'
    ),
}
EMERG_COLORS = {
    'Foundation models / LLMs': '#1f4e79',
    'Federated learning':        '#c0392b',
    'Agents / agentic AI':       '#2e8b57',
    'Model distillation':        '#e07b39',
}

years_e   = list(range(2010, 2024))
by_year_e = genuine.groupby('year').size()
raw_e, norm_e = {}, {}
for label, pattern in EMERGING.items():
    counts = []
    for y in years_e:
        sub  = genuine[genuine['year'] == y]
        mask = sub['reasoning'].fillna('').str.lower().str.contains(pattern, regex=True)
        counts.append(mask.sum())
    raw_e[label]  = counts
    norm_e[label] = [c / by_year_e[y] * 1000 for c, y in zip(counts, years_e)]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
for label, counts in raw_e.items():
    ax1.plot(years_e, counts, color=EMERG_COLORS[label], lw=2,
             marker='o', markersize=4, label=label)
ax1.set_xlabel('Publication year', fontsize=10)
ax1.set_ylabel('Patent mentions (count)', fontsize=10)
ax1.set_title('(a) Absolute mentions in LLM reasoning field', fontsize=10, loc='left')
ax1.legend(fontsize=8.5, frameon=False)
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)
ax1.set_xlim(2010, 2023)
ax1.tick_params(labelsize=8)
ax1.set_xticks(years_e[::2])

for label, counts in norm_e.items():
    ax2.plot(years_e, counts, color=EMERG_COLORS[label], lw=2,
             marker='o', markersize=4, label=label)
ax2.set_xlabel('Publication year', fontsize=10)
ax2.set_ylabel('Mentions per 1,000 genuine AI patents', fontsize=10)
ax2.set_title('(b) Normalised emergence rate', fontsize=10, loc='left')
ax2.legend(fontsize=8.5, frameon=False)
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)
ax2.set_xlim(2010, 2023)
ax2.tick_params(labelsize=8)
ax2.set_xticks(years_e[::2])

plt.tight_layout(pad=1.5)
plt.savefig('figures/fig7_emerging_tech.pdf', bbox_inches='tight')
plt.savefig('figures/fig7_emerging_tech.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Done.")

# ══════════════════════════════════════════════════════════════════════════
# FIG 8 — Regression analysis: each office paired with own-jurisdiction investment
# ══════════════════════════════════════════════════════════════════════════
print("Generating fig8_regression...")

try:
    import statsmodels.formula.api as smf
except ImportError:
    print("  statsmodels not installed — run: pip install statsmodels")
    print("  Skipping fig8.")
else:
    # US private AI investment (Stanford AI Index, billions USD)
    us_investment = {
        2010: 1.7, 2011: 2.1, 2012: 2.8, 2013: 3.7, 2014: 6.9, 2015: 12.7,
        2016: 17.0, 2017: 24.6, 2018: 40.4, 2019: 36.0, 2020: 36.8,
        2021: 93.5, 2022: 47.4, 2023: 67.2,
    }
    # EU-10 AI investment (millions EUR), OECD.AI / Preqin, 2015-2023
    eu10_investment = {
        2015: 29489.83, 2016: 33489.64, 2017: 39014.33, 2018: 46618.71,
        2019: 71881.30, 2020: 85039.71, 2021: 102157.69, 2022: 128565.66,
        2023: 126070.91,
    }
    # EPO genuine AI patent counts by year (from EPO BigQuery + Qwen classification)
    epo_counts = {
        2010: 766, 2011: 833, 2012: 872, 2013: 1043, 2014: 1252, 2015: 1370,
        2016: 1643, 2017: 2009, 2018: 2537, 2019: 3579, 2020: 5173,
        2021: 6398, 2022: 7294, 2023: 8032,
    }

    us_reg = genuine.groupby('year').size().reset_index()
    us_reg.columns = ['year', 'genuine_ai_patents']
    epo_reg = pd.DataFrame({'year': list(epo_counts.keys()),
                             'genuine_ai_patents': list(epo_counts.values())})

    # Restrict both to 2015-2023 for direct comparison
    us_reg = us_reg[us_reg['year'] >= 2015].copy()
    epo_reg = epo_reg[epo_reg['year'] >= 2015].copy()

    us_reg['inv'] = us_reg['year'].map(us_investment)
    epo_reg['inv'] = epo_reg['year'].map(eu10_investment)

    for d in (us_reg, epo_reg):
        d['log_patents'] = np.log(d['genuine_ai_patents'])
        d['log_inv']     = np.log(d['inv'])
        d['covid']       = d['year'].isin([2020, 2021]).astype(int)

    m_us  = smf.ols('log_patents ~ covid + log_inv', data=us_reg).fit()
    m_epo = smf.ols('log_patents ~ covid + log_inv', data=epo_reg).fit()

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    ax = axes[0]
    ax.axvspan(2019.5, 2021.5, color='#f0e0a0', alpha=0.4, zorder=0, label='COVID-19 period')
    ax.plot(us_reg['year'], np.exp(m_us.fittedvalues), color='#1f4e79', lw=2,
            linestyle='--', zorder=2, label='Model fit')
    ax.scatter(us_reg['year'], us_reg['genuine_ai_patents'], color='#1f4e79', s=55,
               zorder=3, label='Observed')
    ax.set_xlabel('Publication year', fontsize=10)
    ax.set_ylabel('USPTO genuine AI patents', fontsize=10)
    ax.set_title(f'(a) USPTO ~ US investment ($R^2$={m_us.rsquared:.3f})', fontsize=10, loc='left')
    ax.legend(fontsize=8, frameon=False, loc='upper left')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.yaxis.set_major_formatter(
        plt.FuncFormatter(lambda x, _: f'{int(x/1000)}k' if x >= 1000 else str(int(x))))
    ax.set_xticks(range(2015, 2024, 2))

    ax = axes[1]
    ax.axvspan(2019.5, 2021.5, color='#f0e0a0', alpha=0.4, zorder=0, label='COVID-19 period')
    ax.plot(epo_reg['year'], np.exp(m_epo.fittedvalues), color='#c0392b', lw=2,
            linestyle='--', zorder=2, label='Model fit')
    ax.scatter(epo_reg['year'], epo_reg['genuine_ai_patents'], color='#c0392b', s=55,
               marker='s', zorder=3, label='Observed')
    ax.set_xlabel('Publication year', fontsize=10)
    ax.set_ylabel('EPO genuine AI patents', fontsize=10)
    ax.set_title(f'(b) EPO ~ EU-10 investment ($R^2$={m_epo.rsquared:.3f})', fontsize=10, loc='left')
    ax.legend(fontsize=8, frameon=False, loc='upper left')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_xticks(range(2015, 2024, 2))

    ax = axes[2]
    labels = ['COVID-19\n(2020-2021)', 'log(own-jurisdiction\nAI investment)']
    us_coefs  = [m_us.params['covid'], m_us.params['log_inv']]
    us_lo     = [m_us.conf_int().loc['covid', 0], m_us.conf_int().loc['log_inv', 0]]
    us_hi     = [m_us.conf_int().loc['covid', 1], m_us.conf_int().loc['log_inv', 1]]
    epo_coefs = [m_epo.params['covid'], m_epo.params['log_inv']]
    epo_lo    = [m_epo.conf_int().loc['covid', 0], m_epo.conf_int().loc['log_inv', 0]]
    epo_hi    = [m_epo.conf_int().loc['covid', 1], m_epo.conf_int().loc['log_inv', 1]]

    y = np.arange(len(labels))
    width = 0.32
    ax.barh(y + width/2, us_coefs, width, color='#1f4e79', alpha=0.85, label='USPTO ~ US inv.')
    ax.errorbar(us_coefs, y + width/2,
                xerr=[np.array(us_coefs) - np.array(us_lo), np.array(us_hi) - np.array(us_coefs)],
                fmt='none', color='#333333', lw=1.3, capsize=3)
    ax.barh(y - width/2, epo_coefs, width, color='#c0392b', alpha=0.85, label='EPO ~ EU inv.')
    ax.errorbar(epo_coefs, y - width/2,
                xerr=[np.array(epo_coefs) - np.array(epo_lo), np.array(epo_hi) - np.array(epo_coefs)],
                fmt='none', color='#333333', lw=1.3, capsize=3)
    ax.axvline(0, color='#333333', lw=0.8, linestyle='--')
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=9.5)
    ax.set_xlabel('Coefficient (log genuine AI patents)', fontsize=10)
    ax.set_title('(c) Own-jurisdiction coefficients\n(95% CI)', fontsize=10, loc='left')
    ax.legend(fontsize=9, frameon=False, loc='lower right')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.tick_params(left=False)

    plt.tight_layout(pad=1.5)
    plt.savefig('figures/fig8_regression.pdf', bbox_inches='tight')
    plt.savefig('figures/fig8_regression.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  Done.")

print("All 8 figures saved to figures/")

# ══════════════════════════════════════════════════════════════════════════
# FIG 9 — Country-level breakdown of genuine AI EPO patents
# Requires: epo_candidates_with_country.csv (from bigquery_epo_with_country.py)
#           merged with epo_results_qwen_bigquery.csv
# ══════════════════════════════════════════════════════════════════════════
print("Generating fig9_country_breakdown...")

import matplotlib.patches as mpatches

# Pre-computed country totals from merged EPO classification + BigQuery country data
# (20,387 genuine AI patents with country identified, 2010-2023)
country_data = [
    ('JP',4478,21.97,49.57),('CN',3593,17.62,72.49),('DE',2369,11.62,33.95),
    ('KR',2069,10.15,29.77),('FR',1085,5.32,12.53),('GB',1028,5.04,12.35),
    ('NL',912,4.47,13.55),('IN',702,3.44,11.46),('SE',595,2.92,9.86),
    ('US',552,2.71,5.95),('CH',467,2.29,7.33),('FI',418,2.05,5.13),
    ('IT',210,1.03,2.44),('ES',207,1.02,2.13),('DK',169,0.83,2.60),
    ('AU',167,0.82,1.91),('BE',159,0.78,2.18),('TW',139,0.68,1.97),
    ('IE',117,0.57,1.23),('AT',77,0.38,0.93),
]
df_c = pd.DataFrame(country_data, columns=['country','total','share','slope'])

EU27 = {'AT','BE','BG','HR','CY','CZ','DK','EE','FI','FR','DE','GR',
        'HU','IE','IT','LV','LT','LU','MT','NL','PL','PT','RO','SK',
        'SI','ES','SE'}
EU10 = {'DE','FR','NL','IT','IE','ES','BE','SE','AT','DK'}

df_c['is_eu27'] = df_c['country'].isin(EU27)
df_c['is_eu10'] = df_c['country'].isin(EU10)

eu10_inv_2023 = {'DE':26629,'FR':30928,'NL':18306,'IT':12049,'IE':7421,
                 'ES':8236,'BE':7042,'SE':6785,'AT':4635,'DK':4039}

fig, axes = plt.subplots(1, 3, figsize=(17, 7))

# Panel A: Top 15 countries
ax = axes[0]
top15 = df_c.head(15).copy()
colors_a = ['#1f4e79' if eu else '#aaaaaa' for eu in top15['is_eu27']]
ax.barh(range(len(top15)), top15['total'], color=colors_a, alpha=0.85)
ax.set_yticks(range(len(top15)))
ax.set_yticklabels(top15['country'], fontsize=9)
ax.invert_yaxis()
ax.set_xlabel('Genuine AI patents filed at EPO (2010–2023)', fontsize=9)
ax.set_title('(a) Top 15 applicant countries', fontsize=10, loc='left')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
eu_patch  = mpatches.Patch(color='#1f4e79', alpha=0.85, label='EU27')
non_patch = mpatches.Patch(color='#aaaaaa', alpha=0.85, label='Non-EU')
ax.legend(handles=[eu_patch, non_patch], fontsize=8, frameon=False, loc='lower right')
ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x,_: f'{int(x/1000)}k' if x>=1000 else str(int(x))))

# Panel B: EU-10 scatter investment vs patents
ax = axes[1]
eu10_df = df_c[df_c['is_eu10']].copy()
eu10_df['inv_2023'] = eu10_df['country'].map(eu10_inv_2023)
ax.scatter(eu10_df['inv_2023'], eu10_df['total'], color='#1f4e79', s=80, alpha=0.85, zorder=3)
for _, row in eu10_df.iterrows():
    ax.annotate(row['country'], xy=(row['inv_2023'], row['total']),
                xytext=(5,3), textcoords='offset points', fontsize=8.5, color='#333333')
x = eu10_df['inv_2023'].values
y = eu10_df['total'].values
m, b = np.polyfit(x, y, 1)
xline = np.linspace(x.min(), x.max(), 100)
ax.plot(xline, m*xline+b, color='#c0392b', lw=1.5, linestyle='--', alpha=0.7)
r = np.corrcoef(x, y)[0,1]
ax.set_xlabel('EU-10 AI investment 2023 (M EUR, OECD.AI)', fontsize=9)
ax.set_ylabel('Total genuine AI patents at EPO 2010–2023', fontsize=9)
ax.set_title(f'(b) EU-10: investment vs EPO patenting\n(r = {r:.2f})', fontsize=10, loc='left')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# Panel C: EU-10 annual line chart
eu10_by_year = {
    'DE': {2010:81,2011:65,2012:42,2013:49,2014:50,2015:48,2016:54,2017:64,2018:72,2019:191,2020:300,2021:414,2022:441,2023:498},
    'FR': {2010:26,2011:31,2012:25,2013:34,2014:40,2015:46,2016:55,2017:79,2018:66,2019:66,2020:91,2021:125,2022:172,2023:229},
    'NL': {2010:14,2011:10,2012:14,2013:21,2014:21,2015:9,2016:19,2017:26,2018:57,2019:108,2020:145,2021:161,2022:156,2023:151},
    'SE': {2010:11,2011:7,2012:14,2013:12,2014:10,2015:12,2016:8,2017:18,2018:28,2019:45,2020:55,2021:88,2022:112,2023:175},
    'IT': {2010:11,2011:7,2012:8,2013:8,2014:4,2015:5,2016:6,2017:8,2018:8,2019:17,2020:14,2021:28,2022:43,2023:43},
    'ES': {2010:2,2011:8,2012:8,2013:11,2014:13,2015:11,2016:7,2017:6,2018:10,2019:14,2020:20,2021:20,2022:45,2023:32},
    'BE': {2010:0,2011:6,2012:2,2013:8,2014:3,2015:6,2016:4,2017:7,2018:8,2019:13,2020:24,2021:22,2022:31,2023:25},
    'DK': {2010:1,2011:6,2012:3,2013:3,2014:5,2015:5,2016:4,2017:4,2018:7,2019:8,2020:23,2021:24,2022:38,2023:38},
    'AT': {2010:1,2011:5,2012:2,2013:1,2014:0,2015:0,2016:6,2017:5,2018:4,2019:7,2020:6,2021:10,2022:10,2023:20},
    'IE': {2010:1,2011:0,2012:4,2013:9,2014:2,2015:6,2016:8,2017:2,2018:9,2019:16,2020:10,2021:18,2022:15,2023:17},
}
years_c = list(range(2010, 2024))
colors_line = {'DE':'#1f4e79','FR':'#2e75b6','NL':'#2e86c1','SE':'#1a9641'}
top4 = ['DE','FR','NL','SE']
rest_c = ['IT','ES','BE','DK','AT','IE']

ax = axes[2]
for country in top4:
    vals = [eu10_by_year[country].get(y,0) for y in years_c]
    ax.plot(years_c, vals, color=colors_line[country], lw=2.2, label=country, zorder=3)
for country in rest_c:
    vals = [eu10_by_year[country].get(y,0) for y in years_c]
    ax.plot(years_c, vals, color='#aaaaaa', lw=1.0, linestyle='--', alpha=0.7, zorder=2)
ax.plot([], [], color='#aaaaaa', lw=1.0, linestyle='--', label='IT, ES, BE, DK, AT, IE')
ax.set_xlabel('Publication year', fontsize=9)
ax.set_ylabel('Genuine AI patents per year', fontsize=9)
ax.set_title('(c) EU-10 annual EPO AI patent filings\nby country, 2010–2023', fontsize=10, loc='left')
ax.legend(fontsize=8.5, frameon=False, loc='upper left')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.set_xticks(range(2010, 2024, 2))

plt.tight_layout(pad=1.5)
plt.savefig('figures/fig9_country_breakdown.pdf', bbox_inches='tight')
plt.savefig('figures/fig9_country_breakdown.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Done.")
print("All 9 figures saved to figures/")
