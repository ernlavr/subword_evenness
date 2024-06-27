import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import gaussian_kde
from scipy.spatial import ConvexHull
import math
import os

def rad2deg(rad):
    return rad * (180 / np.pi)

def get_this_dir():
    return os.path.dirname(os.path.abspath(__file__))

langs_short = ['eng']
langs_full = ['english']

new_results = pd.DataFrame(columns=['Language', 'Angle', 'Mode'])
results = pd.DataFrame(columns=['language', 'k', 'b', 'rad_angle', 'deg_angle'])
new_angle = pd.DataFrame(columns=['Var1', 'Var2', 'Freq', 'lang'])
new_data_fun = pd.DataFrame(columns=['x', 'values', 'diagonals', 'lang'])

angles = [0] * 100

for l in range(len(langs_short)):
    print(langs_short[l])
    output_dir = os.path.join(get_this_dir(), '..', '..', 'output', langs_full[l])
    results_path = os.path.join(output_dir, f"{langs_full[l]}_full_results.csv")
    df = pd.read_csv(results_path, sep='\t')
    
    df = df[df['segments_lengths'].str.contains(',')]
    
    lang_short = langs_short[l]
    lang_full = langs_full[l]
    
    for iter in range(100):
        df1 = df[df['language'] == lang_full]
        freqs = df1.groupby(['index', 'word_length']).size().reset_index(name='Freq')
        freqs = freqs[freqs['Freq'] != 0]
        freqs['lang'] = lang_short
        
        freqs['Var1'] = freqs['index'].astype(int)
        freqs['Var2'] = freqs['word_length'].astype(int)
        
        freqs = freqs[['Var1', 'Var2', 'Freq']]
        
        sns.set(style="whitegrid")
        plt.figure(figsize=(9, 5))
        p = sns.scatterplot(x='Var2', y='Var1', size='Freq', data=freqs, legend=False, sizes=(20, 200))
        plt.title(f'{lang_full}, {lang_short}')
        plt.xlabel('word length in characters')
        plt.ylabel('unevenness index')
        
        # Density plot
        xy = np.vstack([freqs['Var2'], freqs['Var1']])
        z = gaussian_kde(xy)(xy)
        idx = z.argsort()
        x, y, z = freqs['Var2'].values[idx], freqs['Var1'].values[idx], z[idx]
        p = plt.scatter(x, y, c=z, s=100, edgecolor='red', cmap='Blues')

        plt.colorbar(p)
        
        # Extracting max points
        sub = freqs.copy()
        max_y = sub['Var1'].max()
        max_x = sub[sub['Var1'] == max_y]['Var2'].iloc[0]
        
        max_right_x = sub['Var2'].max()
        max_right_y = sub[sub['Var2'] == max_right_x]['Var1'].iloc[0]
        
        x1, y1 = max_x, max_y
        x2, y2 = max_right_x, max_right_y
        
        X = np.array([[x1, 1], [x2, 1]])
        y = np.array([y1, y2])
        coef = np.linalg.solve(X, y)
        
        k, b = coef
        
        theta1 = np.arctan(1)
        theta2 = np.arctan(k)
        rad_angle = np.pi - abs(theta1 - theta2)
        deg_angle = rad2deg(rad_angle)
        
        def left_diagonal(x):
            return x - 2
        
        def right_diagonal(x):
            return k * x + b
        
        x_vals = np.linspace(0, 100, 400)
        plt.plot(x_vals, left_diagonal(x_vals), color='red')
        plt.plot(x_vals, right_diagonal(x_vals), color='green')
        
        data_fun = pd.DataFrame({
            'x': np.concatenate((x_vals, x_vals)),
            'values': np.concatenate((left_diagonal(x_vals), right_diagonal(x_vals))),
            'diagonals': ['left diagonal'] * len(x_vals) + ['right diagonal'] * len(x_vals)
        })
        data_fun['lang'] = lang_short
        
        plt.title(f'{lang_full}\nBPE-Min-R')
        plt.xlim(0, 30)
        plt.ylim(0, 30)
        plt.xlabel('word length in characters')

        # Save plot
        sue_dir = os.path.join(output_dir, 'sue_plots')
        os.makedirs(sue_dir, exist_ok=True)
        plt.savefig(os.path.join(sue_dir, f"{lang_full}_{iter}.png"), bbox_inches='tight')
        plt.close()
        
        # Assuming 'results' is your existing DataFrame and the variables are defined
        new_row = pd.DataFrame({'language': [lang_full], 'k': [k], 'b': [b], 'rad_angle': [rad_angle], 'deg_angle': [deg_angle]})
        results = pd.concat([results, new_row], ignore_index=True)   
             
        freqs['lang'] = f'{lang_full}, {lang_short}'
        new_angle = pd.concat([new_angle, freqs], ignore_index=True)
        
        data_fun['lang'] = f'{lang_full}, {lang_short}'
        new_data_fun = pd.concat([new_data_fun, data_fun], ignore_index=True)
        
        angles[iter] = deg_angle
    
    def closest(xv, sv):
        return xv[np.argmin(np.abs(np.array(xv) - sv))]
    
    mean_angle = np.mean(angles)
    closest_angle = closest(angles, mean_angle)
    index = np.argmin(np.abs(np.array(angles) - mean_angle))
    
    print(lang_full)
    print(index)
    print(mean_angle)

    # Replacing append with concat
    new_row = pd.DataFrame([{'Language': lang_full, 'Angle': mean_angle, 'Mode': 'BPE-Min-R'}])
    new_results = pd.concat([new_results, new_row], ignore_index=True)


new_results = new_results.dropna()
print(new_results)
