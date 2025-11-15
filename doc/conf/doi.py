"""
Helper script to generate the doi.rst file with the Zenodo DOI corresponding
to current PyAbel version (if possible, otherwise the default "latest" DOI).
"""

ver = 'v' + release
doi_file = 'doi.rst'
try:
    # check current file
    with open(doi_file, 'r') as f:
        if ' ' + ver + '\n' not in f.read():
            raise Exception('DOI file needs updating.')
    print('DOI file is OK.')
except Exception as e:
    print(e)
    print('Retrieving current DOI from Zenodo...')
    doi = '10.5281/zenodo.594858'  # default (concept) DOI
    try:
        from urllib.request import urlopen
        import json
        # get info for all versions from Zenodo API
        # (TODO: can we just get the specific version?)
        data = json.load(urlopen('https://zenodo.org/api/records/'
                                 '?q=conceptrecid:594858'
                                 '&all_versions=True&sort=-version&size=9999'))
        # search for given version
        for rec in data['hits']['hits']:
            md = rec['metadata']
            if md['version'] == ver:
                doi = md['doi']
                break
        else:
            raise LookupError(f'Error: Version {ver} not found.')
        print(f'  Found DOI {doi} for version {ver}.')
    except Exception as e:
        print(f'  {e}\n  Using concept DOI {doi}.')
        ver = 'PyAbel'  # don't use specific version
    # write file
    with open(doi_file, 'w') as f:
        f.write(f'''\
..
    DOI for {ver}
    Don't edit this file, it's updated automatically by conf.py!

.. |doi| replace:: {doi}

.. |doi_link| replace:: `{doi} <https://doi.org/{doi}>`__
''')
