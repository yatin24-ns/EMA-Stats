import pandas as pd
import hashlib
import matplotlib.pyplot as plt
import tkinter as tk #newlib
from tkinter import ttk

#DS initialisation and definitions 
INPUT  = r"C:\Users\praka\OneDrive\Desktop\CHINTA practice\EMA consent form and Trait-behaviour (Responses).xlsx"
OUTPUT = r"C:\Users\praka\OneDrive\Desktop\CHINTA practice\output.xlsx"
FIG1   = r"C:\Users\praka\OneDrive\Desktop\CHINTA practice\fig1_demographics.png"
FIG2   = r"C:\Users\praka\OneDrive\Desktop\CHINTA practice\fig2_fatigue.png"

LIKERT = {"doesn't apply at all": 1, "does not apply much": 2,
          "slightly applies": 3, "applies a lot": 4, "applies completely": 5}

EDU_RANK = {"no formal education": 1,
            "high school diploma or equivalent (10th/12th)": 2,
            "associate degree or vocational training": 3,
            "bachelor's degree": 4, "master's degree": 5,
            "doctorate or professional degree (e.g., phd, md, jd)": 6}

AGE_BINS   = [0, 20, 25, 30, 35, 40, 50, 100]
AGE_LABELS = ['<=20', '21-25', '26-30', '31-35', '36-40', '41-50', '50+']
PAL = ['#4C72B0', '#DD8452', '#55A868', '#C44E52', '#8172B3', '#937860', '#DA8BC3']
 
#Step1- Load the data 
def load(path):
    #Load Excel, keep only consenting participants 
    df = pd.read_excel(path)
    n = len(df)
    ok = lambda c: df.iloc[:, c].astype(str).str.strip().str.lower().str.startswith('yes')
    df = df[ok(1) & ok(2)].reset_index(drop=True)
    print(f"Loaded {n} -> {len(df)} consenting")
    return df if len(df) else None

#Step2- Apply SHA-256 to hash names 
def anonymise(df, name_idx=3):
    nc = df.columns[name_idx]
    df.insert(0, 'P_No', [f"P{i:03d}" for i in range(1, len(df) + 1)])
    df.insert(1, 'P_ID', df[nc].apply(
        lambda x: int(hashlib.sha256(str(x).strip().lower().encode()).hexdigest(), 16) % 10**8
        if pd.notna(x) else None))
    return df

def process(path):
    df = load(path)
    if df is None:
        return None

    age_c, gen_c, edu_c = df.columns[5], df.columns[6], df.columns[7]
    lik_c = list(df.columns[8:18])
    df = anonymise(df)

    #Age
    df[age_c] = pd.to_numeric(df[age_c], errors='coerce')
    df['Age_Group'] = pd.cut(df[age_c], AGE_BINS, labels=AGE_LABELS, right=True)
    q1, q3 = df[age_c].quantile(.25), df[age_c].quantile(.75)
    age_stats = pd.DataFrame({'Metric': ['Mean','Median','SD','Min','Max','Q1','Q3','IQR','Skew'],
                              'Value': [df[age_c].mean(), df[age_c].median(), df[age_c].std(),
                                        df[age_c].min(), df[age_c].max(), q1, q3, q3-q1,
                                        df[age_c].skew()]}).round(2)
    #Gender
    def pg(r):
        if pd.isna(r): return 'Other'
        return {'m':'Male','f':'Female'}.get(str(r).strip().lower()[0], 'Other')
    df['Gender'] = df[gen_c].apply(pg)

    #Education
    df['Education'] = df[edu_c].astype(str).str.strip().str.lower()
    df['Edu_Rank']  = df['Education'].map(EDU_RANK)

    #Likert scoring
    qn = [f"Q{i}" for i in range(1, 11)]
    df[qn] = df[lik_c].astype(str).apply(lambda c: c.str.strip().str.lower()).apply(lambda c: c.map(LIKERT))
    df['Fat_Total']  = df[qn].sum(axis=1)
    df['Fat_Mean']   = df[qn].mean(axis=1)
    df['Fat_SD']     = df[qn].std(axis=1)
    df['Extreme_5s'] = (df[qn] == 5).sum(axis=1)

    #Risk flag: participants >= 1-SD+mean 
    cohort_mean = df['Fat_Total'].mean()
    cohort_sd   = df['Fat_Total'].std()
    df['High_Risk'] = df['Fat_Total'] >= (cohort_mean + cohort_sd)

    #table
    def freq(col):
        t = df[col].value_counts().reset_index()
        t.columns = [col, 'N']
        t['%'] = (t['N'] / t['N'].sum() * 100).round(1)
        return t

    # per-demographic fatigue breakdown
    rows = []
    for fac, col in [('Gender','Gender'), ('Age Group','Age_Group'), ('Education','Education')]:
        for lv, s in df.groupby(col, observed=False)['Fat_Total']:
            lbl = str(lv).title() if fac == 'Education' else str(lv)
            rows.append({'Factor': fac, 'Level': lbl, 'N': len(s),
                         'Mean': round(s.mean(), 2), 'SD': round(s.std(), 2),
                         'Median': round(s.median(), 2),
                         'High_Risk_N': int((s >= cohort_mean + cohort_sd).sum())})
    combined = pd.DataFrame(rows)

    tables = {
        'age_stats':  age_stats,
        'age_dist':   freq('Age_Group').sort_values('Age_Group').reset_index(drop=True),
        'gen_dist':   freq('Gender'),
        'gen_age':    pd.crosstab(df['Gender'], df['Age_Group']),
        'edu_dist':   freq('Education'),
        'fat_stats':  pd.DataFrame({
            'Metric': ['Cohort Mean','SD','Median','Min','Max','High-Risk Count'],
            'Value':  [round(cohort_mean,2), round(cohort_sd,2),
                       round(df['Fat_Total'].median(),2),
                       df['Fat_Total'].min(), df['Fat_Total'].max(),
                       int(df['High_Risk'].sum())]
        }),
        'item_means': df[qn].mean().round(2).to_frame('Mean'),
        'combined':   combined,
    }
    return df, tables, age_c, qn


#print
def show(title, tbl):
    print(f"\n{'='*60}\n  {title}\n{'='*60}")
    print(tbl.to_string(index='item' in title.lower()) if isinstance(tbl, pd.DataFrame) else tbl)

def print_report(T):
    titles = [('AGE STATISTICS', 'age_stats'), ('AGE GROUP DISTRIBUTION', 'age_dist'),
              ('GENDER DISTRIBUTION', 'gen_dist'), ('GENDER x AGE CROSSTAB', 'gen_age'),
              ('EDUCATION DISTRIBUTION', 'edu_dist'), ('FATIGUE PSYCHOMETRICS', 'fat_stats'),
              ('PER-ITEM MEAN SCORES', 'item_means'), ('COMBINED EFFECTS TABLE', 'combined')]
    for title, key in titles: show(title, T[key])


# Figure 1:
def plot_demographics(df, T, age_col):
    fig, ax = plt.subplots(2, 2, figsize=(11, 8.5))
    fig.suptitle('Cohort Demographics', fontsize=14, weight='bold', y=0.98)

    # A. Age
    a = df[age_col].dropna()
    ax[0,0].hist(a, bins=10, color=PAL[0], edgecolor='white', label='Cohort Age')
    ax[0,0].axvline(a.mean(), color='red', ls='--', label=f'Mean={a.mean():.1f}')
    ax[0,0].set_ylim(0, ax[0,0].get_ylim()[1] * 1.25) # Headroom for legend

    # B. Gender
    g = T['gen_dist']
    bars = ax[0,1].bar(g['Gender'], g['N'], color=PAL[1:1+len(g)], edgecolor='white', width=0.5)
    for b in bars:
        ax[0,1].text(b.get_x()+b.get_width()/2, b.get_height()+.2, int(b.get_height()), ha='center', fontsize=9, weight='bold')
    ax[0,1].set_ylim(0, ax[0,1].get_ylim()[1] * 1.20) # Headroom for labels

    # C. Education
    e = T['edu_dist'].sort_values('N')
    labs = [x.title()[:25]+'..' if len(x)>25 else x.title() for x in e['Education']]
    ax[1,0].barh(labs, e['N'], color=PAL[4], edgecolor='white', height=0.5)
    ax[1,0].set_xlim(0, ax[1,0].get_xlim()[1] * 1.15) # Headroom for horizontal bars

    # D. Fatigue histogram
    f = df['Fat_Total'].dropna()
    ax[1,1].hist(f, bins=10, color=PAL[3], edgecolor='white', label='Fatigue Total')
    ax[1,1].axvline(f.mean(), color='navy', ls='--', label=f'Mean={f.mean():.1f}')
    ax[1,1].set_ylim(0, ax[1,1].get_ylim()[1] * 1.25) # Headroom for legend

    for axis, xl, yl, tit, has_legend, loc in [
        (ax[0,0], 'Age (yrs)', 'Count (N)', 'A. Age Distribution', True, 'upper right'),
        (ax[0,1], '', 'Count (N)', 'B. Gender Distribution', False, 'upper right'),
        (ax[1,0], 'Count (N)', '', 'C. Education Level Distribution', False, 'lower right'),
        (ax[1,1], 'Total Fatigue Score (10-50)', 'Count (N)', 'D. Fatigue Score Distribution', True, 'upper right')
    ]:
        if xl: axis.set_xlabel(xl, labelpad=8)
        if yl: axis.set_ylabel(yl, labelpad=8)
        axis.set_title(tit, pad=10, weight='bold')
        if has_legend:
            axis.legend(fontsize=8, loc=loc)
            axis.grid(axis='y', alpha=.25)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(FIG1, dpi=200, bbox_inches='tight')
    return fig


#Cross-demographic Fatigue
def plot_fatigue(df, T, qcols):
    fig, ax = plt.subplots(2, 2, figsize=(11, 8.5))
    fig.suptitle('Fatigue Across Demographics', fontsize=14, weight='bold', y=0.98)

    def bplot(ax, col, title, color, max_lbl=0):
        gd = df.dropna(subset=[col, 'Fat_Total']).groupby(col)['Fat_Total']
        data  = [g.values for _, g in gd]
        names = [str(n) for n, _ in gd]
        if max_lbl:
            names = [n.title()[:max_lbl]+'..' if len(n)>max_lbl else n.title() for n in names]
        if not data: return
        bp = ax.boxplot(data, labels=names, patch_artist=True, widths=.4)
        for b in bp['boxes']: b.set(facecolor=color, alpha=.7)
        ax.set(ylabel='Fatigue Score', title=title)
        ax.set_ylim(5, 55) # Clean bounds for fatigue score (range is 10-50)
        ax.grid(axis='y', alpha=.25)

    for axis, col, tit, clr, lbl in [
        (ax[0,0], 'Gender', 'A. Fatigue by Gender', PAL[1], 0),
        (ax[0,1], 'Age_Group', 'B. Fatigue by Age Group', PAL[0], 0),
        (ax[1,0], 'Education', 'C. Fatigue by Education', PAL[4], 16)
    ]: 
        bplot(axis, col, tit, clr, lbl)
        axis.set_title(tit, pad=10, weight='bold')
        axis.set_ylabel('Fatigue Score', labelpad=8)
    
    ax[0,1].tick_params(axis='x', rotation=30)
    ax[1,0].tick_params(axis='x', rotation=30, labelsize=8)

    # D. Item means
    means = df[qcols].mean()
    ax[1,1].bar(range(1,11), means.values, color=PAL[5], edgecolor='white', width=0.6, label='Mean Score')
    ax[1,1].axhline(3, color='gray', ls='--', alpha=.5, label='Neutral (3.0)')
    ax[1,1].set(xlabel='Item #', ylabel='Mean Score (1-5)', title='D. Per-Item Means', xticks=range(1,11))
    ax[1,1].set_ylim(0, 6.0) # Headroom for the legend
    ax[1,1].set_title('D. Per-Item Means', pad=10, weight='bold')
    ax[1,1].set_xlabel('Item #', labelpad=8)
    ax[1,1].set_ylabel('Mean Score (1-5)', labelpad=8)
    ax[1,1].legend(fontsize=8, loc='upper right')
    ax[1,1].grid(axis='y', alpha=.25)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(FIG2, dpi=200, bbox_inches='tight')
    return fig


def make_table_view(parent, title, tbl):
    frame = ttk.LabelFrame(parent, text=title, padding=10)
    text = tk.Text(frame, wrap=tk.NONE, height=min(len(tbl) + 3, 12), font=("Consolas", 10), bd=0, highlightthickness=0)
    
    show_index = False
    if isinstance(tbl, pd.DataFrame):
        if tbl.index.name is not None or not isinstance(tbl.index, pd.RangeIndex) or 'crosstab' in title.lower() or 'item' in title.lower() or 'stats' in title.lower():
            show_index = True
        text.insert(tk.END, tbl.to_string(index=show_index))
    else:
        text.insert(tk.END, str(tbl))
        
    text.configure(state=tk.DISABLED)
    xscroll = ttk.Scrollbar(frame, orient=tk.HORIZONTAL, command=text.xview)
    text.configure(xscrollcommand=xscroll.set)
    text.pack(fill=tk.BOTH, expand=True)
    xscroll.pack(fill=tk.X)
    return frame

def launch_gui_dashboard(df, T, age_col, qcols, fig1, fig2):
    import tkinter as tk
    from tkinter import ttk
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

    root = tk.Tk()
    root.title("Clinical Data Pipeline & Fatigue Dashboard")
    root.geometry("1400x900")


    # Main Scrollable Setup
    main_frame = ttk.Frame(root)
    main_frame.pack(fill=tk.BOTH, expand=True)

    canvas = tk.Canvas(main_frame)
    scrollbar_y = ttk.Scrollbar(main_frame, orient=tk.VERTICAL, command=canvas.yview)
    scrollbar_x = ttk.Scrollbar(main_frame, orient=tk.HORIZONTAL, command=canvas.xview)
    scrollable_frame = ttk.Frame(canvas)

    scrollable_frame.bind(
        "<Configure>",
        lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
    )

    canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
    canvas.configure(yscrollcommand=scrollbar_y.set, xscrollcommand=scrollbar_x.set)

    scrollbar_y.pack(side=tk.RIGHT, fill=tk.Y)
    scrollbar_x.pack(side=tk.BOTTOM, fill=tk.X)
    canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

    # Enable mousewheel scrolling on canvas
    def _on_mousewheel(event):
        # Windows handles event.delta
        canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

    def _on_shift_mousewheel(event):
        canvas.xview_scroll(int(-1 * (event.delta / 120)), "units")
    
    canvas.bind_all("<MouseWheel>", _on_mousewheel)
    canvas.bind_all("<Shift-MouseWheel>", _on_shift_mousewheel)

    #Demographic Section
    row1 = ttk.LabelFrame(scrollable_frame, text="1. Cohort Demographics", padding=15)
    row1.pack(fill=tk.X, padx=15, pady=10, expand=True)

    #Left column for tables, Right column for Plot
    left_tables_1 = ttk.Frame(row1)
    left_tables_1.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 15))

    for title, key in [("Age Statistics", 'age_stats'), ("Age Group Distribution", 'age_dist'),
                       ("Gender Distribution", 'gen_dist'), ("Gender x Age Crosstab", 'gen_age'),
                       ("Education Distribution", 'edu_dist')]:
        make_table_view(left_tables_1, title, T[key]).pack(fill=tk.X, pady=5)

    right_plot_1 = ttk.Frame(row1)
    right_plot_1.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

    canvas_fig1 = FigureCanvasTkAgg(fig1, master=right_plot_1)
    canvas_fig1.draw()
    canvas_fig1.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    #Fatigue & Psychometrics Section 
    row2 = ttk.LabelFrame(scrollable_frame, text="2. Fatigue & Psychometrics", padding=15)
    row2.pack(fill=tk.X, padx=15, pady=10, expand=True)

    #Left column for tables, Right column for Plot
    left_tables_2 = ttk.Frame(row2)
    left_tables_2.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 15))

    for title, key in [("Fatigue Statistics Summary", 'fat_stats'), ("Per-Item Mean Scores (Q1 - Q10)", 'item_means'),
                       ("Combined Demographic Effects on Fatigue", 'combined')]:
        make_table_view(left_tables_2, title, T[key]).pack(fill=tk.X, pady=5)

    right_plot_2 = ttk.Frame(row2)
    right_plot_2.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

    canvas_fig2 = FigureCanvasTkAgg(fig2, master=right_plot_2)
    canvas_fig2.draw()
    canvas_fig2.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    # Clean up figure objects when closing the window
    def on_closing():
        plt.close('all')
        root.destroy()
    root.protocol("WM_DELETE_WINDOW", on_closing)
    root.mainloop()

def main():
    result = process(INPUT)
    if result is None:
        print("No consenting participants. Exiting."); return

    df, T, age_col, qcols = result
    print_report(T)
    fig1 = plot_demographics(df, T, age_col)
    fig2 = plot_fatigue(df, T, qcols)
    df.to_excel(OUTPUT, index=False)
    print(f"\nDone -> {OUTPUT}")
    
    # Launch scrollable side-by-side Tkinter dashboard
    launch_gui_dashboard(df, T, age_col, qcols, fig1, fig2)

if __name__ == '__main__':
    main()