import pandas as pd
import statsmodels.api as sm

from collections import defaultdict
from scipy import stats
from statsmodels.formula.api import glm
from utils import normalize_column_names


def cast_to_numeric(df, numerical_predictors):
    for col in df.columns:
        df.loc[df[col]=='unclear', col] = 0
        df.loc[df[col].isna(), col] = 0
        df[col] = pd.to_numeric(df[col], downcast='unsigned')
        # Remove invalid values in categorical variables
        if col not in numerical_predictors:
            unique_values = df[col].unique()
            invalid_values = set(unique_values) - set([0,1])
            if len(invalid_values) > 0:  # the column has invalid values                   
                invalid_indexes = list(df.loc[~df[col].isin([0,1])].index)
                if len(invalid_indexes) > 0:
                    print(f'Found the following invalid values {invalid_values} '\
                          f'in {len(invalid_indexes)} rows of the column {col}. '\
                          f'Rows has been discarded.')
                    df = df.drop(index=invalid_indexes)
    return df


def check_corr_num_vars(df, numerical_predictors, alpha_level):
    num_var_1 = numerical_predictors[0]
    num_var_2 = numerical_predictors[1]
    _, p_val = stats.spearmanr(
        df[num_var_1], 
        df[num_var_2]
    )
    if p_val < alpha_level:
        return True
    else:
        return False


def check_corr_cat_num_vars(df, numerical_predictors, predictors, alpha_level):
    independence_num_cat = pd.DataFrame(columns=numerical_predictors, 
                                        index=predictors)
    independence_num_cat_vars = defaultdict(list)
    for predictor_1 in numerical_predictors:
        for predictor_2 in predictors:
            if predictor_2 == predictor_1:
                continue
            _, p_val = stats.spearmanr(df[predictor_1],df[predictor_2])        
            if p_val < alpha_level:
                independence_num_cat.loc[predictor_2,predictor_1] = 'F'
            else:
                independence_num_cat.loc[predictor_2,predictor_1] = 'T'
                independence_num_cat_vars[predictor_1].append(predictor_2)
    return independence_num_cat_vars


def cast_to_category(df, numerical_predictors):
    for col in df.columns:
        if col not in numerical_predictors:
            df[col] = df[col].astype('category')
    return df


def check_corr_cat_vars(df, predictors, numerical_predictors, alpha_level):
    independece_analysis = pd.DataFrame(columns=predictors, 
                                        index=predictors)
    for predictor_1 in predictors:
        if predictor_1 in numerical_predictors:
            continue
        for predictor_2 in predictors:        
            if predictor_2 in numerical_predictors:
                continue
            if predictor_2 == predictor_1:
                continue
            cross_tab = pd.crosstab(df[predictor_1], df[predictor_2])
            _, p_val, _, _ = stats.chi2_contingency(cross_tab)
            if p_val < alpha_level:
                independece_analysis.loc[predictor_1,predictor_2] = 'F'
            else:
                independece_analysis.loc[predictor_1,predictor_2] = 'T'
    return independece_analysis


def obtain_possible_models(independece_analysis, independence_num_cat_vars, \
                           num_vars_correlated):
    possible_models = [None]*len(independece_analysis.columns)
    idx = 0
    for col in independece_analysis.columns:
        possible_models[idx] = [col]
        for row in independece_analysis.index:
            if independece_analysis.loc[row, col] == 'T':
                possible_models[idx].append(row)
        # Add independent numerical variables
        for num_var, ind_cat_vars in independence_num_cat_vars.items():        
            if col in ind_cat_vars:
                possible_models[idx].append(num_var)                
                if num_vars_correlated:
                    # If numerical variables are correlated,
                    # after adding a numerical variable to the model
                    # we break loop because both numerical variables 
                    # cannot be together in a model since they are
                    # not independent
                    break
        idx += 1
    return possible_models


def create_formulas_for_possible_models(possible_models, numerical_predictors):
    models = []
    for possible_model in possible_models:
        formula = f'disagreement ~ '
        num_predictors = len(possible_model)
        for idx, predictor in enumerate(possible_model):
            if predictor not in numerical_predictors:
                formula += f' C({predictor})'
            else:
                formula += f' {predictor}'
            if idx < (len(possible_model)-1):
                formula += ' + '
        models.append({'formula': formula, 'num_predictors': num_predictors, 
                       'predictors': possible_model})
    return models


def fit_models(models, df):
    idx_not_fitted_models = []
    for idx, model_dict in enumerate(models):
        try:
            formula = model_dict['formula']
            model = glm(formula, data = df, family = sm.families.Binomial()).fit()
            model_dict['model'] = model
            model_dict['deviance'] = model.deviance
            model_dict['log_likelihood'] = model.llf 
        except Exception as e:
            idx_not_fitted_models.append(idx)
            print(f'Error: {e}.\nModel:\n{formula}')
            print()
    return idx_not_fitted_models


def update_unfitted_models(idx_not_fitted_models, possible_models, 
                           numerical_predictors, df):
    updated_models = []
    for idx_not_fitted_model in idx_not_fitted_models:
        predictors = possible_models[idx_not_fitted_model].copy()
        for predictor in predictors:
            ind_vars = possible_models[idx_not_fitted_model].copy()        
            ind_vars.remove(predictor)
            formula = 'disagreement ~ '
            for idx, ind_var in enumerate(ind_vars):
                if ind_var not in numerical_predictors:
                    formula += f' C({ind_var})'
                else:
                    formula += f' {ind_var}'
                if idx < (len(ind_vars)-1):
                    formula += ' + '
            try:
                model = glm(formula, data = df, family = sm.families.Binomial()).fit()
                print(f'Variable {predictor} has been removed from the predictors list and the model could be fitted.\n')
                updated_models.append(
                    {
                        'formula': formula, 
                        'model': model, 
                        'num_predictors': len(ind_vars),
                        'predictors': ind_vars,
                        'deviance': model.deviance,
                        'log_likelihood': model.llf
                    }
                )
            except:
                pass
    return updated_models


def select_best_models(models):
    min_dev = 1000000
    best_models = []
    for model in models:
        if 'deviance' in model and model['deviance'] < min_dev:
            min_dev = model['deviance']
            best_models = [model]
        elif 'deviance' in model and model['deviance'] == min_dev:
            best_models.append(model)
    return best_models


def predict_disagreement(df, categorial_predictors, numerical_predictors, theme=None):
    alpha_level = 0.05

    predictors = categorial_predictors + numerical_predictors
    # 1. Select columns that include predictors and the target variable
    if theme:
        p_df = df.loc[df['topic']==theme, predictors+['disagreement']]
    else:
        p_df = df.loc[:,predictors+['disagreement']]
    print(f"The analysis is conducted with a dataset composed of "\
          f"{p_df.shape[0]} rows and {p_df.shape[1]} columns")
    # 2. Cast variables to numeric
    p_df = cast_to_numeric(p_df, numerical_predictors)
    print(f"After the step above the dataset is composed of {p_df.shape[0]} "\
          f"rows and {p_df.shape[1]} columns")
    #3. Check independence between numerical variables
    num_vars_correlated = check_corr_num_vars(p_df, numerical_predictors, alpha_level)
    # 4. Check independence between numerical and categorical variables
    independence_num_cat_vars = check_corr_cat_num_vars(p_df, numerical_predictors, 
                                                        predictors, alpha_level)
    # 5. Cast categorical variables to category type
    p_df = cast_to_category(p_df, numerical_predictors)
    # 6. Check independence of categorial variables
    independece_analysis = check_corr_cat_vars(p_df, predictors, 
                                               numerical_predictors, alpha_level)
    independece_analysis.drop(columns=numerical_predictors, 
                              index=numerical_predictors, inplace=True)
    # 7. Fit models based on independent predictors
    possible_models = obtain_possible_models(independece_analysis, 
                                             independence_num_cat_vars,
                                             num_vars_correlated)
    print(f'There are {len(possible_models)} possible models')
    models = create_formulas_for_possible_models(possible_models, numerical_predictors)
    idx_not_fitted_models = fit_models(models, p_df)
    updated_models = update_unfitted_models(idx_not_fitted_models, possible_models,
                                            numerical_predictors, p_df)
    models.extend(updated_models)
    # 8. Select the best model
    best_models = select_best_models(models)
    print(f"There are {len(best_models)} best model(s)")
    return best_models


def get_dataset():
    member_df = pd.read_csv('data/member_decision_making_and_right.csv')
    admin_df = pd.read_csv('data/administration_of_association.csv')
    informal_df = pd.read_csv('data/informally_organized_group_of_a.csv')
    member_df.columns = normalize_column_names(member_df.columns)
    admin_df.columns = normalize_column_names(admin_df.columns)
    informal_df.columns = normalize_column_names(informal_df.columns)
    member_df = member_df.rename(columns={
        'annotations_for_disagreement_new_idea_cascade_s': 'annotations_for_disagreement_new_idea_cascades'
    })
    admin_df = admin_df.rename(columns={
        'amount_of_likes': 'number_of_likes',
        'comment\'s_id': 'comment_id',
        'proposal': 'proposals',    
    })
    informal_df = informal_df.rename(columns={
        'annotations_for_disagreement_new_idea_cascade_s': 'annotations_for_disagreement_new_idea_cascades',
        'proposal': 'proposals'
    })
    member_df['topic'] = 'member'
    admin_df['topic'] = 'admin'
    informal_df['topic'] = 'informal'
    all_df = pd.concat([member_df, admin_df, informal_df], axis=0, ignore_index=True)
    all_df = all_df.drop(['comment', 'response'], axis=1)
    all_df = all_df.rename(columns={'comment_1': 'comment', 'response_1': 'response'})
    return all_df


if __name__ == "__main__":
    all_df = get_dataset() 
    predictors_disagreement = [
        'number_of_likes', 'number_of_ideas', 'simple_disagreement', 'elaborated_disagreement',
        'topic_shift', 'brainstorming', 'blending', 'building', 'broadening', 'fact',
        'value', 'policy', 'interpretation', 'gives_reason_s', 'presents_evidence', 'asks_question_s',
        'provides_information', 'clarifies_position_stance', 'responds_to_previous_comment', 
        'constructive_tone', 'moderator_post', 'acknowledges_problem'
    ]  
    numerical_predictors = ['number_of_likes', 'number_of_ideas']
    categorical_predictors = predictors_disagreement.copy()
    for numerical_predictor in numerical_predictors:
        categorical_predictors.remove(numerical_predictor)
    best_models = predict_disagreement(all_df, categorical_predictors, numerical_predictors, 'informal')
    print(best_models[0]['model'].summary())
    