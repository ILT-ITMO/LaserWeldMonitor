
def check_gosts(item):
    gosts = []
    t = item['t']
    if item['hm'] < min(0.1 * t, 2):
        gosts.append((507, 'B'))
    elif item['hm'] < min(0.15 * t, 2):
        gosts.append((507, 'C'))
    elif item['hm'] < min(0.25 * t, 3):
        gosts.append((507, 'D'))
    else:
        gosts.append((507, 'None'))

    if item['hg'] < min(0.1 * t, 0.5):
        gosts.append((515, 'B'))
    elif item['hg'] < min(0.2 * t, 0.5):
        gosts.append((515, 'C'))
    elif item['hg'] < min(0.3 * t, 1):
        gosts.append((515, 'D'))
    else:
        gosts.append((515, 'None'))

    if item['hp'] < min(0.1 * t, 0.5):
        gosts.append((504, 'B'))
    elif item['hp'] < min(0.2 * t, 0.5):
        gosts.append((504, 'C'))
    elif item['hp'] < min(0.3 * t, 1):
        gosts.append((504, 'D'))
    else:
        gosts.append((504, 'None'))

    if item['hs'] < min(0.1 * t, 0.5):
        gosts.append((509, 'B'))
    elif item['hs'] < min(0.2 * t, 0.5):
        gosts.append((509, 'C'))
    elif item['hs'] < min(0.3 * t, 1):
        gosts.append((509, 'D'))
    else:
        gosts.append((509, 'None'))

    if item['hs'] < min(0.05 * t, 0.5):
        gosts.append((5011, 'B'))
    elif item['hs'] < min(0.1 * t, 0.5):
        gosts.append((5011, 'C'))
    elif item['hs'] < min(0.15 * t, 1):
        gosts.append((5011, 'D'))
    else:
        gosts.append((5011, 'None'))

    if item['he'] < min(0.2 + 0.15 * t, 5):
        gosts.append((502, 'B'))
    elif item['he'] < min(0.2 + 0.2 * t, 5):
        gosts.append((502, 'C'))
    elif item['he'] < min(0.2 + 0.3 * t, 5):
        gosts.append((502, 'D'))
    else:
        gosts.append((502, 'None'))

    gosts_dict = dict(gosts)
    print(gosts_dict)

    gosts_dict['key'] = item['key']
    return gosts_dict