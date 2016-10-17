import pickle
import pandas as pd
import math
import os
import sys
import operator
import matplotlib.pyplot as plt
sys.path.append("../tools")
from parse_out_email_text import parseOutText
from feature_format import featureFormat

def load_dict(fn):
    with open(fn, "r") as data_file:
        data_dict = pickle.load(data_file)
    return data_dict

def remove_outliers(data_dict):
    '''
    Remove unnecessary data.
    Total and The Travel Agency in the Park do not refer to people.
    Eugene has no values and is also removed.
    '''
    del(data_dict['TOTAL'])
    del(data_dict['THE TRAVEL AGENCY IN THE PARK'])
    del(data_dict['LOCKHART EUGENE E'])

    return data_dict

def remove_rows_limited_features(data_dict):
    '''remove people with only 2 features'''
    del(data_dict['WODRASKA JOHN'])
    del(data_dict['WHALEY DAVID A'])
    del(data_dict['WROBEL BRUCE'])
    del(data_dict['SCRIMSHAW MATTHEW'])
    del(data_dict['GRAMM WENDY L'])

    return data_dict

def update_bad_finance_values(data_dict):
    '''  Updates the financial data for 'BELFER ROBERT' and 'BHATNAGAR SANJAY' to the correct values.

    :param data_dict: dictionary of features/values for the Enron project
    :return: updated data_dict
    '''
    data_dict['BELFER ROBERT'].update({'salary': 'NaN',
                                  'bonus': 'NaN',
                                  'long_term_incentive': 'NaN',
                                  'deferred_income': -102500,
                                  'deferral_payments': 'NaN',
                                  'loan_advances': 'NaN',
                                  'other': 'NaN',
                                  'expenses': 3285,
                                  'director_fees': 102500,
                                  'total_payments': 3285,
                                  'exercised_stock_options': 'NaN',
                                  'restricted_stock': 44093,
                                  'restricted_stock_deferred': -44093,
                                  'total_stock_value': 'NaN'})

    data_dict['BHATNAGAR SANJAY'].update({'salary': 'NaN',
                                  'bonus': 'NaN',
                                  'long_term_incentive': 'NaN',
                                  'deferred_income': 'NaN',
                                  'deferral_payments': 'NaN',
                                  'loan_advances': 'NaN',
                                  'other': 'NaN',
                                  'expenses': 137864,
                                  'director_fees': 'NaN',
                                  'total_payments': 137864,
                                  'exercised_stock_options': 15456290,
                                  'restricted_stock': 2604490,
                                  'restricted_stock_deferred': -2604490,
                                  'total_stock_value': 15456290})

    return data_dict

def get_emails(data_dict, direction='from'):
    ''' Retrieves the persons that have any e-mails sent from them

    :param data_dict:
    :param direction: get email to or from the person
    :return: List of names of people with e-mails that are from them
    '''

    # def check_duplicates(matches):
    # '''check for any duplicate matches.  This indicated multiple names are matching.
    #     # check for any duplicates
    #     for i in range(len(matches)):
    #         for j in range(i + 1, len(matches)):
    #             if matches[i][0] == matches[j][0]:
    #                 print matches[i][0], matches[i][1], matches[j][1]

    def remove_mismatch(matches):
        '''My match function is simplistic and returned some incorret matches.  Remove these.'''
        mismatch_list = ['from_e.taylor@enron.com.txt', 'from_eric.le@enron.com.txt', 'from_john.oh@enron.com.txt']
        res = []
        for match in matches:
            if match[1] not in mismatch_list:
                res.append(match)
            else:
                print "removing", match[1], "from", match[0]
        return res

    import re
    names = data_dict.keys()
    files = os.listdir("emails_by_address")
    direction += '_'
    matches = []
    for f in files:
        if direction in f:
            author = f.lstrip(direction)
            author = author.split("@")[0]
            full_name = author.split(".")
            if len(full_name) == 2:
                first_name = full_name[0]
                last_name = full_name[1]

                for name in names:
                    if first_name.upper() in name and last_name.upper() in name:
                        matches.append((name, f))

    matches = remove_mismatch(matches)
    return matches

def add_word_ratios(data_dict, matches):
    ''' Creates new features with word count ratios for the given words.
    Updates the data_dict with these new features

    :param data_dict:
    :param matches:
    :return: updated data_dict with word ratios
    '''
    results = {}

    for name, filename in matches:
        print name, filename
        word_counts = {#'concern':0,
                      #'blame':0,
                      #'problem':0,
                      #'assur':0,
                      'enron':0,
                      'team':0,
                      'want':0,
                      'let':0,
                      'veri':0,
                      'issu':0,
                      'provid':0,
                      'depreci':0
                      }
        with open("emails_by_address/" + filename) as f:
            n = 0
            for email in f.readlines():
                n += 1
                path = "/".join(email.strip().split("/")[1:])
                with open("../" + path) as ff:
                    words = parseOutText(ff)
                    for word in words.split():
                        for w in word_counts:
                            if w in word:
                                word_counts[w] += 1
            #normalize to # emails
            for word in word_counts:
                word_counts[word] /= float(n)

        print word_counts
        results[name] = word_counts

    for name in data_dict:
        blank = {}
        for word in word_counts:
            blank[word] = "NaN"
        data_dict[name].update(blank)
    for name in results:
        data_dict[name].update(results[name])

    return data_dict

def filter_poi(data_dict, matches):
    ''' Takes a list of mathces and returns only those that are POIs'''
    filtered_matches = []
    for name in matches:
        if data_dict[name[0]]['poi'] == True:
            filtered_matches.append(name)
    return filtered_matches

def get_word_counts(txt):
    ''' Returns a list of the top 200 words found in a given set of e-mails

    :param txt: .txt file with list a of e-mails files
    :return: top 200 words found in all e-mails listed under the .txt file, ignoring stop words
    '''
    sw = [u'i', u'me', u'my', u'myself', u'we', u'our', u'ours', u'ourselves', u'you', u'your', u'yours', u'yourself',
     u'yourselves', u'he', u'him', u'his', u'himself', u'she', u'her', u'hers', u'herself', u'it', u'its', u'itself',
     u'they', u'them', u'their', u'theirs', u'themselves', u'what', u'which', u'who', u'whom', u'this', u'that',
     u'these', u'those', u'am', u'is', u'are', u'was', u'were', u'be', u'been', u'being', u'have', u'has', u'had',
     u'having', u'do', u'does', u'did', u'doing', u'a', u'an', u'the', u'and', u'but', u'if', u'or', u'because', u'as',
     u'until', u'while', u'of', u'at', u'by', u'for', u'with', u'about', u'against', u'between', u'into', u'through',
     u'during', u'before', u'after', u'above', u'below', u'to', u'from', u'up', u'down', u'in', u'out', u'on', u'off',
     u'over', u'under', u'again', u'further', u'then', u'once', u'here', u'there', u'when', u'where', u'why', u'how',
     u'all', u'any', u'both', u'each', u'few', u'more', u'most', u'other', u'some', u'such', u'no', u'nor', u'not',
     u'only', u'own', u'same', u'so', u'than', u'too', u'very', u's', u't', u'can', u'will', u'just', u'don', u'should',
     u'now'] #from nltk english stopwords
    word_counts = {}
    with open("emails_by_address/" + txt) as f:
        for email in f.readlines():
            path = "/".join(email.strip().split("/")[1:])
            with open("../" + path) as ff:
                words = parseOutText(ff)
                for word in words.split():
                    if word not in sw:
                        if word not in word_counts:
                            word_counts[word] = 1
                        else:
                            word_counts[word] += 1

    sorted_counts = sorted(word_counts.items(), key=operator.itemgetter(1), reverse=True)
    return sorted_counts[:200]

def add_message_fractions(data_dict):
    '''
    Creates new dictionary key:values for the ratio of messages to/from POIs.
    Returns an updated data dictionary with the new features
    '''

    for name in data_dict:
        data = data_dict[name]

        #to messages ratio
        if data['to_messages'] == 'NaN':
            data['to_poi_ratio'] = 'NaN'
        else:
            to_messages = float(data['to_messages'])
            to_POI = data['from_this_person_to_poi']
            if to_POI == 'NaN':
                to_POI = 0.
            else:
                to_POI = float(to_POI)
            data['to_poi_ratio'] = to_POI / to_messages

        # from messages ratio
        if data['from_messages'] == 'NaN':
            data['from_poi_ratio'] = 'NaN'
        else:
            from_messages = float(data['from_messages'])
            from_POI = data['from_poi_to_this_person']
            if from_POI == 'NaN':
                from_POI = 0.
            else:
                from_POI = float(to_POI)
            data['from_poi_ratio'] = from_POI / from_messages

    return data_dict

def add_scaled_financial(data_dict):
    '''  Scales certain financial features and adds to the data_dict

    :param data_dict:
    :return: updated data_dict with new scaled features
    '''

    features = ['bonus', 'exercised_stock_options']

    for feature in features:
        for name in data_dict:
            #sqrt
            if data_dict[name][feature] == 'NaN':
                data_dict[name][feature + '_sqrt'] = 'NaN'
            else:
                data_dict[name][feature + '_sqrt'] = math.sqrt(float(data_dict[name][feature]))
            #log
            if data_dict[name][feature] == 'NaN':
                data_dict[name][feature + '_log'] = 'NaN'
            else:
                data_dict[name][feature + '_log'] = math.log10(float(data_dict[name][feature]))

    return data_dict

def make_scatter(data_dict, feature1, feature2, color_poi = True):
    '''Make a scatter plot for quick feature comparison
    Color codes the poi feature for visual analysis'''
    data = featureFormat(data_dict, ['poi'] + [feature1, feature2])
    data = pd.DataFrame(data = data, columns = ['poi', feature1, feature2])
    data.plot(x=feature1, y=feature2, kind='scatter', c='poi')
    plt.show()

def dump_data_dict(data_dict):
    with open('my_dataset_test.pkl', "w") as dataset_outfile:
        pickle.dump(data_dict, dataset_outfile)

if __name__ == "__main__":
    fn = "final_project_dataset.pkl"
    data_dict = load_dict(fn)
    data_dict = remove_outliers(data_dict)
    data_dict = add_scaled_financial(data_dict)
    data_dict = add_message_fractions(data_dict)
    data_dict = add_word_ratios(data_dict, get_emails(data_dict))
    dump_data_dict(data_dict)