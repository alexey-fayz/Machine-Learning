import numpy as np
import pandas as pd
from scipy import stats

def task1_numpy_arrays(matrix_file='../data/matrix.npy'):
    """
    Р—Р°РґР°РЅРёРµ 1: Р Р°Р±РѕС‚Р° СЃ NumPy РјР°СЃСЃРёРІР°РјРё
    """
    matrix = np.load(matrix_file)
    
    total_sum = round(matrix.sum(), 2)
    mean_val = round(matrix.mean(), 2)
    max_val = round(matrix.max(), 2)
    min_val = round(matrix.min(), 2)
    diag_sum = round(np.trace(matrix), 2)  # СЃСѓРјРјР° РіР»Р°РІРЅРѕР№ РґРёР°РіРѕРЅР°Р»Рё
    
    # РїРѕР±РѕС‡РЅР°СЏ РґРёР°РіРѕРЅР°Р»СЊ
    n = matrix.shape[0]
    anti_diag_sum = round(sum(matrix[i, n-1-i] for i in range(n)), 2)
    
    return {
        'sum': total_sum,
        'mean': mean_val,
        'max': max_val,
        'min': min_val,
        'diag_sum': diag_sum,
        'anti_diag_sum': anti_diag_sum
    }


def task2_dataframe_basic(data_file='../data/data_pd.csv'):
    """
    Р—Р°РґР°РЅРёРµ 2: Р‘Р°Р·РѕРІС‹Рµ РѕРїРµСЂР°С†РёРё СЃ DataFrame
    """
    df = pd.read_csv(data_file)
    
    mean_age = round(df['age'].mean(), 2)
    median_income = round(df['income'].median(), 2)
    phd_count = (df['education'] == 'PhD').sum()
    max_experience = round(df['experience'].max(), 2)
    mean_performance = round(df['performance'].mean(), 3)
    it_count = (df['department'] == 'IT').sum()
    income_std = round(df['income'].std(), 2)
    exp_income_corr = round(df['experience'].corr(df['income']), 3)
    high_satisfaction_percent = round((df['satisfaction'] >= 4).mean() * 100, 2)
    
    return {
        'mean_age': mean_age,
        'median_income': median_income,
        'phd_count': phd_count,
        'max_experience': max_experience,
        'mean_performance': mean_performance,
        'it_count': it_count,
        'income_std': income_std,
        'exp_income_corr': exp_income_corr,
        'high_satisfaction_percent': high_satisfaction_percent
    }


def task3_groupby_aggregation(data_file='../data/data_pd.csv'):
    """
    Р—Р°РґР°РЅРёРµ 3: Р“СЂСѓРїРїРёСЂРѕРІРєР° Рё Р°РіСЂРµРіР°С†РёСЏ
    """
    df = pd.read_csv(data_file)
    
    # РЎСЂРµРґРЅРёР№ РґРѕС…РѕРґ РїРѕ РѕС‚РґРµР»Р°Рј
    avg_income_by_dept = df.groupby('department')['income'].mean().round(2).to_dict()
    # РњР°РєСЃРёРјР°Р»СЊРЅС‹Р№ Р±РѕРЅСѓСЃ РїРѕ СѓСЂРѕРІРЅСЏРј РѕР±СЂР°Р·РѕРІР°РЅРёСЏ
    max_bonus_by_education = df.groupby('education')['bonus'].max().round(2).to_dict()
    # РљРѕР»РёС‡РµСЃС‚РІРѕ СЃРѕС‚СЂСѓРґРЅРёРєРѕРІ РїРѕ РѕС‚РґРµР»Р°Рј
    count_by_dept = df.groupby('department').size().to_dict()
    # РЎСЂРµРґРЅРёР№ РѕРїС‹С‚ РїРѕ РѕС‚РґРµР»Р°Рј
    avg_exp_by_dept = df.groupby('department')['experience'].mean().round(2).to_dict()
    # РњРµРґРёР°РЅРЅР°СЏ РїСЂРѕРёР·РІРѕРґРёС‚РµР»СЊРЅРѕСЃС‚СЊ РїРѕ РѕС‚РґРµР»Р°Рј
    median_performance_by_dept = df.groupby('department')['performance'].median().round(3).to_dict()
    # РЎС‚Р°РЅРґР°СЂС‚РЅРѕРµ РѕС‚РєР»РѕРЅРµРЅРёРµ РґРѕС…РѕРґР° РїРѕ СѓСЂРѕРІРЅСЏРј РѕР±СЂР°Р·РѕРІР°РЅРёСЏ
    income_std_by_education = df.groupby('education')['income'].std().round(2).to_dict()
    # РљРѕСЌС„С„РёС†РёРµРЅС‚ РІР°СЂРёР°С†РёРё РІРѕР·СЂР°СЃС‚Р° РїРѕ РѕС‚РґРµР»Р°Рј
    age_cv_by_dept = (df.groupby('department')['age'].std() / df.groupby('department')['age'].mean()).round(3).to_dict()
    
    return {
        'avg_income_by_dept': avg_income_by_dept,
        'max_bonus_by_education': max_bonus_by_education,
        'count_by_dept': count_by_dept,
        'avg_exp_by_dept': avg_exp_by_dept,
        'median_performance_by_dept': median_performance_by_dept,
        'income_std_by_education': income_std_by_education,
        'age_cv_by_dept': age_cv_by_dept
    }


def task4_data_filtering(data_file='../data/data_pd.csv'):
    """
    Р—Р°РґР°РЅРёРµ 4: Р¤РёР»СЊС‚СЂР°С†РёСЏ РґР°РЅРЅС‹С…
    """
    df = pd.read_csv(data_file)
    
    mean_income_over_30 = round(df[df['age'] > 30]['income'].mean(), 2)
    max_bonus_it = round(df[df['department'] == 'IT']['bonus'].max(), 2)
    mean_exp_phd = round(df[df['education'] == 'PhD']['experience'].mean(), 2)
    high_income_count = (df['income'] > 100000).sum()
    mean_satisfaction_finance = round(df[df['department'] == 'Finance']['satisfaction'].mean(), 2)
    
    mean_exp_overall = df['experience'].mean()
    above_avg_exp_it_count = len(df[(df['department'] == 'IT') & (df['experience'] > mean_exp_overall)])
    
    performance_std_young = round(df[df['age'] < 25]['performance'].std(), 3)
    
    # РСЃРїСЂР°РІР»РµРЅРѕ: РїСЂРѕС†РµРЅС‚ РѕС‚ РѕР±С‰РµРіРѕ С‡РёСЃР»Р° СЃРѕС‚СЂСѓРґРЅРёРєРѕРІ
    high_perf_master_percent = round(((df['education'] == 'Master') & (df['performance'] > 0.7)).mean() * 100, 2)
    
    hr_df = df[df['department'] == 'HR']
    age_satisfaction_corr_hr = round(hr_df['age'].corr(hr_df['satisfaction']), 3) if len(hr_df) > 1 else np.nan
    
    master_income = df[df['education'] == 'Master']['income'].mean()
    bachelor_income = df[df['education'] == 'Bachelor']['income'].mean()
    master_bachelor_income_diff = round(master_income - bachelor_income, 2)
    
    return {
        'mean_income_over_30': mean_income_over_30,
        'max_bonus_it': max_bonus_it,
        'mean_exp_phd': mean_exp_phd,
        'high_income_count': high_income_count,
        'mean_satisfaction_finance': mean_satisfaction_finance,
        'above_avg_exp_it_count': above_avg_exp_it_count,
        'performance_std_young': performance_std_young,
        'high_perf_master_percent': high_perf_master_percent,
        'age_satisfaction_corr_hr': age_satisfaction_corr_hr,
        'master_bachelor_income_diff': master_bachelor_income_diff
    }

def task5_sorting_ranking(data_file='../data/data_pd.csv'):
    """
    Р—Р°РґР°РЅРёРµ 5: РЎРѕСЂС‚РёСЂРѕРІРєР° Рё СЂР°РЅР¶РёСЂРѕРІР°РЅРёРµ
    """
    df = pd.read_csv(data_file)
    
    top_income = df.nlargest(5, 'income')['income'].round(2).tolist()
    top_experience = df.nlargest(5, 'experience')['experience'].tolist()
    top_performance = df.nlargest(5, 'performance')['performance'].round(3).tolist()
    top_bonus = df.nlargest(5, 'bonus')['bonus'].round(2).tolist()
    
    return {
        'top_income': top_income,
        'top_experience': top_experience,
        'top_performance': top_performance,
        'top_bonus': top_bonus
    }


def task6_income_statistics(data_file='../data/data_pd.csv'):
    """
    Р—Р°РґР°РЅРёРµ 6: Р’С‹С‡РёСЃР»РµРЅРёРµ СЃС‚Р°С‚РёСЃС‚РёРє
    """
    df = pd.read_csv(data_file)
    income = df['income']
    
    mean_income = round(income.mean(), 2)
    median_income = round(income.median(), 2)
    q1 = income.quantile(0.25)
    q3 = income.quantile(0.75)
    income_iqr = round(q3 - q1, 2)
    income_cv = round(income.std() / income.mean(), 3)
    income_geom_mean = round(stats.gmean(income), 2)
    income_log_mean = round(np.mean(np.log(income)), 3)
    
    return {
        'mean_income': mean_income,
        'median_income': median_income,
        'income_iqr': income_iqr,
        'income_cv': income_cv,
        'income_geom_mean': income_geom_mean,
        'income_log_mean': income_log_mean
    }


def task7_bernoulli_distribution(bernoulli_file='../data/bernoulli.npy'):
    """
    Р—Р°РґР°РЅРёРµ 7: Р Р°СЃРїСЂРµРґРµР»РµРЅРёРµ Р‘РµСЂРЅСѓР»Р»Рё
    """
    bernoulli_sample = np.load(bernoulli_file)
    
    p_hat = round(bernoulli_sample.mean(), 3)
    # P(X >= 60) РґР»СЏ Binomial(100, p_hat)
    p_at_least_60 = round(1 - stats.binom.cdf(59, 100, p_hat), 3)
    
    return {
        'p_hat': p_hat,
        'p_at_least_60': p_at_least_60
    }


def task8_poisson_distribution(poisson_file='../data/poisson.npy'):
    """
    Р—Р°РґР°РЅРёРµ 8: Р Р°СЃРїСЂРµРґРµР»РµРЅРёРµ РџСѓР°СЃСЃРѕРЅР°
    """
    poisson_sample = np.load(poisson_file)
    
    lambda_hat = round(poisson_sample.mean(), 2)
    p_x_equals_3 = round(stats.poisson.pmf(3, lambda_hat), 3)
    p_x_greater_5 = round(1 - stats.poisson.cdf(5, lambda_hat), 3)
    
    return {
        'lambda_hat': lambda_hat,
        'p_x_equals_3': p_x_equals_3,
        'p_x_greater_5': p_x_greater_5
    }


def task9_exponential_distribution(exponential_file='../data/exponential.npy'):
    """
    Р—Р°РґР°РЅРёРµ 9: РРєСЃРїРѕРЅРµРЅС†РёР°Р»СЊРЅРѕРµ СЂР°СЃРїСЂРµРґРµР»РµРЅРёРµ
    """
    exp_sample = np.load(exponential_file)
    
    lambda_hat = round(1 / exp_sample.mean(), 3)
    p_x_greater_15 = round(np.exp(-lambda_hat * 15), 3)
    
    return {
        'lambda_hat': lambda_hat,
        'p_x_greater_15': p_x_greater_15
    }