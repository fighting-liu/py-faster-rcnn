from ConfigParser import SafeConfigParser

def config(**kwargs):
    if kwargs.get('conf_file', None) is None:
        return kwargs
    parser = SafeConfigParser()
    parser.read(kwargs.get('conf_file'))
    # print parser.sections()

    conf = {}
    if 'section_name' in kwargs:
        sec_name = kwargs['section_name']
    else:
        raise 'No section name specified'


    if parser.has_section(sec_name):
        for option, value in parser.items(sec_name):
            conf[option] = value
    elif 'section_name' in kwargs:
        raise 'No section name found'

    return conf


if __name__ == '__main__':
    conf_file = 'detection.cfg'    
    section_name = 'handbag_train'
    conf = config(section_name=section_name, conf_file=conf_file)

    for key, val in conf.iteritems():
        print key, type(val)


