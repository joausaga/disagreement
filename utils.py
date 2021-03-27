import pandas as pd

def normalize_column_names(names):
    names = names.str.lower()
    names = names.str.replace(' ','_')
    names = names.str.replace('.','_')
    names = names.str.replace('(','_')
    names = names.str.replace(')','')
    names = names.str.replace('/','_')
    names = names.str.replace('___','_')
    names = names.str.strip()
    return names

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
    # Remove summary rows
    idxs_to_remove = all_df[all_df.background.isnull()].index.values
    all_df = all_df.drop(index=idxs_to_remove)
    # Fix errors in rows 336 and 310 in which simple_agreement and 
    # elaborated_agreement were incorrectly annotated
    all_df.loc[336, 'simple_agreement'] = 0
    all_df.loc[310, 'elaborated_agreement'] = 1
    return all_df