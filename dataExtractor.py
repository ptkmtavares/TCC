from email.parser import HeaderParser
import re
import email.utils as emailUtils
import random

parser = HeaderParser()

features = [
    'time_zone',
    'date_date_received_diff',
    'missing_importance',
    'missing_x-mailman-version',
    'missing_user-agent',
    'missing_x-mailer',
    'missing_list-unsubscribe',
    'missing_mime-version',
    'missing_references',
    'missing_x-original-to',
    'missing_domainkey-signature',
    'missing_received-spf',
    'missing_dmarc',
    'str_content-encoding_empty',
    'str_from_question',
    'str_from_exclam',
    'str_from_chevron',
    'str_to_chevron',
    'str_to_undisclosed',
    'str_to_empty',
    'str_message-ID_dollar',
    'str_return-path_bounce',
    'str_return-path_empty',
    'str_reply-to_question',
    'str_content-type_texthtml',
    'str_precedence_list',
    'str_received-SPF_bad',
    'str_received-SPF_softfail',
    'str_received-SPF_fail',
    'str_dmarc_bad',
    'str_dmarc_softfail',
    'str_dmarc_fail',
    'str_dkim_bad',
    'str_dkim_softfail',
    'str_dkim_fail',
    'received_str_forged'
    ]

header_information = [
    'label',
    'received_hop_1',
    'received_hop_2',
    'received_hop_3',
    'received_hop_4',
    'received_hop_5',
    'received_hop_6',
    'received_hop_7',
    'received_hop_8',
    'received_hop_9',
    'received_hop_10',
    'received_hop_11',
    'received_hop_12',
    'received_hop_13',
    'received_hop_14',
    'received_hop_15',
    'received_hop_16',
    'from',
    'date',
    'hop_count',
    'subject',
    'message-id',
    'to',
    'content-type',
    'mime-version',
    'x-mailer',
    'content-transfer-encoding',
    'x-mimeole',
    'x-priority',
    'return-path',
    'list-id',
    'lines',
    'x-virus-scanned',
    'status',
    'reply-To', 
    'content-length',
    'precedence',
    'delivered-to',
    'list-unsubscribe',
    'list-subscribe',
    'list-post',
    'list-help',
    'x-msmail-priority',
    'x-spam-status',
    'sender',
    'errors-to',
    'reply-to',
    'x-beenthere',
    'list-archive',
    'x-mailman-version',
    'x-miltered',
    'x-uuid',
    'x-virus-status',
    'x-spam-level',
    'x-spam-checker-version',
    'references',
    'user-agent',
    'received-spf',
    'in-reply-to',
    'x-original-to',
    'user-agent',
    'arc-message-signature',
    'arc-authentication-results',
    'arc-seal',
    'thread-index',
    'cc',
    'content-disposition',
    'mailing-list',
    'x-spam-check-by',
    'domainkey-signature',
    'dkim-signature',
    'importance',
    'x-mailing-list'
    ]

def getStrFeatures(email_info, info_name, feature_list, features, conditions_check):
    for feature_name, condition_check in zip(features, conditions_check):
        if condition_check == '':
            if re.match(condition_check, email_info[info_name]):
                feature_list[feature_name] = 1
            else:
                feature_list[feature_name] = 0
        else:
            if re.search(condition_check, email_info[info_name], re.IGNORECASE):
                feature_list[feature_name] = 1
            else:
                feature_list[feature_name] = 0

def getTimeZone(email_info):
    time_zone = emailUtils.parsedate_tz(email_info['date'])
    if time_zone is None:
        return -1
    else:
        if (int(time_zone[9]/(60*60)) % 24) == 20:
            return 0
        return 1

def getDateDateReceivedDiff(email_info):
    date = emailUtils.parsedate_tz(email_info['date'])
    last_received = email_info['received_hop_' + str(email_info['hop_count'])]
    last_received_list = re.split(r';', last_received)
    last_received_date = emailUtils.parsedate_tz(last_received_list[-1])
    
    if date is None or last_received_date is None:
        return -1
    
    try:
        emailUtils.mktime_tz(date)
        emailUtils.mktime_tz(last_received_date)
    except:
        return -1
    
    date_delta = int((emailUtils.mktime_tz(last_received_date) - emailUtils.mktime_tz(date)))
    if date_delta < 0:
        return 0
    return 1
    
def getMissingFeatures(email_info, feature_list):
    for name in email_info.keys():
        if 'missing_' + name in features:
            if email_info[name] == '':
                feature_list['missing_' + name] = 1
            else: 
                feature_list['missing_' + name] = 0
    # Will change later
    feature_list['missing_dmarc'] = 0

def getReceivedStrForged(email_info):
    n = email_info['hop_count']
    for i in range(1, n+1):
        received = email_info['received_hop_' + str(i)]
        if 'forged' in received:
            return 1
    return 0

def checkEmailType(index_path, filename):
    with open(index_path, 'r', encoding='latin_1') as f:
        for line in f:
            if line.endswith(filename):
                if line.startswith('ham'):
                    return 1
                elif line.startswith('spam'):
                    return 2
                else:
                    return 3

def getFeaturesArray(email_info):
    features_dict = {}
    
    features_dict['time_zone'] = getTimeZone(email_info)
    features_dict['date_comp_date_received'] = getDateDateReceivedDiff(email_info)
    getMissingFeatures(email_info, features_dict)
    
    if (email_info['arc-authentication-results'] != '') and ('spf=none' not in email_info['arc-authentication-results']):
        features_dict['missing_received-spf'] = 0
        
    if (email_info['arc-authentication-results'] != '') and ('dmarc=none' not in email_info['arc-authentication-results']):
        features_dict['missing_dmarc'] = 0
        
    if (email_info['arc-authentication-results'] != '') and ('dkim=none' not in email_info['arc-authentication-results']):
        features_dict['missing_domainkey-signature'] = 0
        
    if (email_info['dkim-signature'] != ''):
        features_dict['missing_domainkey-signature'] = 0
    
    info_name = 'content-transfer-encoding'
    features_names = ['str_content-encoding_empty']
    conditions_check = ['']
    getStrFeatures(email_info, info_name, features_dict, features_names, conditions_check)
    
    info_name = 'from'
    features_names = ['str_from_question', 'str_from_exclam', 'str_from_chevron']
    conditions_check = ['\\?', '!', '<.+>']
    getStrFeatures(email_info, info_name, features_dict, features_names, conditions_check)
    
    info_name = 'to'
    features_names = ['str_to_chevron', 'str_to_undisclosed', 'str_to_empty']
    conditions_check = ['<.+>', 'Undisclosed Recipients', '']
    getStrFeatures(email_info, info_name, features_dict, features_names, conditions_check)
    
    info_name = 'message-id'
    features_names = ['str_message-ID_dollar']
    conditions_check = ['\\$']
    getStrFeatures(email_info, info_name, features_dict, features_names, conditions_check)
    
    info_name = 'return-path'
    features_names = ['str_return-path_bounce', 'str_return-path_empty']
    conditions_check = ['bounce', '']
    getStrFeatures(email_info, info_name, features_dict, features_names, conditions_check)
    
    info_name = 'reply-to'
    features_names = ['str_reply-to_question']
    conditions_check = ['\\?']
    getStrFeatures(email_info, info_name, features_dict, features_names, conditions_check)
    
    info_name = 'received-spf'
    features_names = ['str_received-SPF_bad', 'str_received-SPF_softfail', 'str_received-SPF_fail']
    conditions_check = ['bad', 'softfail', 'fail']
    getStrFeatures(email_info, info_name, features_dict, features_names, conditions_check)
    
    info_name = 'content-type'
    features_names = ['str_content-type_texthtml']
    conditions_check = ['text/html']
    getStrFeatures(email_info, info_name, features_dict, features_names, conditions_check)
    
    info_name = 'precedence'
    features_names = ['str_precedence_list']
    conditions_check = ['list']
    getStrFeatures(email_info, info_name, features_dict, features_names, conditions_check)
    
    info_name = 'arc-authentication-results'
    features_names = ['str_received-SPF_bad', 'str_received-SPF_softfail', 'str_received-SPF_fail']
    conditions_check = ['spf=bad', 'spf=softfail', 'spf=fail']
    getStrFeatures(email_info, info_name, features_dict, features_names, conditions_check)
     
    info_name = 'arc-authentication-results'
    features_names = ['str_dmarc_bad', 'str_dmarc_softfail', 'str_dmarc_fail']
    conditions_check = ['dmarc=bad', 'dmarc=softfail', 'dmarc=fail']
    getStrFeatures(email_info, info_name, features_dict, features_names, conditions_check)
    
    info_name = 'arc-authentication-results'
    features_names = ['str_dkim_bad', 'str_dkim_softfail', 'str_dkim_fail']
    conditions_check = ['dkim=bad', 'dkim=softfail', 'dkim=fail']
    getStrFeatures(email_info, info_name, features_dict, features_names, conditions_check)
    
    features_dict['received_str_forged'] = getReceivedStrForged(email_info)
    
    features_array = []
    for feature in features_dict.keys():
        features_array.append(features_dict[feature])
    
    return features_array
    
def getEmailInfo(email_path):
    email_dict = {}
    for column in header_information:
        email_dict[column] = ''
    try:
        email = open(email_path, 'r', encoding='latin_1').read()
    except:
        return -1
    parser = HeaderParser()
    header = parser.parsestr(email)
    features_lower_case = [x.lower() for x in header.keys()]
    received_list = header.get_all('received')
    hops = 0
    if received_list is not None:
            hops = len(received_list)
            for i, received_field in enumerate(received_list):
                email_dict['received_hop_' + str(i+1)] = received_field
    temp_dict = dict(zip(features_lower_case, header.values()))
    for key in temp_dict.keys():
        if key in header_information:
            email_dict[key] = temp_dict[key]
    email_dict['hop_count'] = hops
    return email_dict  

def getTrainingTestSet(index_path, values, percent):
    train_set = []
    labels = []
    dict = {'ham': 0, 'spam': 1, 'phishing': 2}
    index = open(index_path, 'r', encoding='latin_1').read()
    lines = index.splitlines()
    for line in random.sample(lines, int(len(lines)*percent)):
        line_split = line.split(' ')
        email_path = line_split[1]
        email_info = getEmailInfo(email_path)
        if line_split[0] in values and email_info != -1:
            train_set.append(getFeaturesArray(email_info))
            labels.append(dict[line_split[0]])
    return train_set, labels