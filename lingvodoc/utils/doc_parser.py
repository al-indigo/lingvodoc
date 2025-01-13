import json
import re

from uniparser_erzya import ErzyaAnalyzer
from uniparser_meadow_mari import MeadowMariAnalyzer
from uniparser_udmurt import UdmurtAnalyzer
from uniparser_moksha import MokshaAnalyzer
from uniparser_komi_zyrian import KomiZyrianAnalyzer
from nltk.tokenize import RegexpTokenizer
from hfst_dev import HfstTransducer
from lxml.html import fromstring
from pdb import set_trace as A
import csv
import os
import tempfile
import bs4
import requests
import io

def print_to_str(*args, **kwargs):

    output = io.StringIO()
    print(*args, file=output, **kwargs)
    contents = output.getvalue()
    output.close()
    return contents


span_id_counter = 0
def generate_html_wrap(word, ana_tag_list, lang="", extra_state=""):

    json_list = list()
    for ana_tag in ana_tag_list:
       attr_dict = dict()
       is_unparsed = True
       for attr in ana_tag.attrs:
           attr_dict[attr] = ana_tag[attr]
           if (ana_tag[attr]):
               is_unparsed = False
       if (is_unparsed):
           return word
       json_list.append(attr_dict)

    global span_id_counter
    span_id_counter += 1
    wrap = f"<span class=\"unverified {extra_state}\"" + " id=" + str(span_id_counter) + ">"
    for attr_json in json_list:
        span_id_counter += 1
        encoded_attrs = ((json.dumps(attr_json, ensure_ascii=False)).encode('utf8')).decode()
        wrap += f"<span class=\"result {extra_state}\"" + " id=" + str(span_id_counter) + ">" + encoded_attrs + "</span>"

        if lang == 'udm' and 'nom' in encoded_attrs:
            flag = True
            span_id_counter += 1
            wrap += f"<span class=\"result {extra_state}\"" + " id=" + str(span_id_counter) + ">" + encoded_attrs.replace('nom', 'acc0') + "</span>"

    wrap += word + "</span>"
    return wrap

def insert_parser_output_to_text(text, parser_output, lang=""):

    ESC_PAT = "$id$"
    soup = bs4.BeautifulSoup(parser_output, 'html.parser')
    w_tag_list = soup("w")
    search_start_index = 0
    result_list = []
    for w_tag in w_tag_list:
        word = w_tag.contents[-1]
        match_index = text.find(word, search_start_index)
        if match_index - len(ESC_PAT) > 0 and match_index + len(word) + len(ESC_PAT) < len(text):
            if text[match_index-len(ESC_PAT):match_index] == ESC_PAT and text[match_index+len(word):match_index+len(word)+len(ESC_PAT)] == ESC_PAT:
                continue
        result_list.append(text[search_start_index:match_index])
        if len(w_tag.contents) > 1:
            extra_state = "broken" if any([a.get('gr') == "Unknown" for a in w_tag.find_all('ana')]) else ""
            result_list.append(generate_html_wrap(word, w_tag.contents[0:-1], lang=lang, extra_state=extra_state))
        search_start_index = match_index + len(word)
    result_list.append(text[search_start_index:])
    result = "".join(result_list)

    return result


def timarkh_uniparser(dedoc_output, lang, has_disamb=False, disambiguate=False):

    wordlist = list()
    tokenizer = RegexpTokenizer(r'(?!\w+(?:-\w+)+)\w+|\w+(?:-\w+)+')
    composite_words = dict()
    i = 0
    for word in tokenizer.tokenize(re.sub(r"(<.*?>)|nbsp", " ", dedoc_output)):
        wordlist.append(word)
        if '-' in word:
            composite_words[i] = word
        i += 1

    if lang == 'udm':
        analyzer = UdmurtAnalyzer(mode='strict')
    if lang == 'erzya':
        analyzer = ErzyaAnalyzer(mode='strict')
    if lang == 'moksha':
        analyzer = MokshaAnalyzer(mode='strict')
    if lang == 'komi_zyryan':
        analyzer = KomiZyrianAnalyzer(mode='strict')
    if lang == 'meadow_mari':
        analyzer = MeadowMariAnalyzer(mode='strict')

    composite_words_values = list(composite_words.values())
    if has_disamb:
        composite_words_output = analyzer.analyze_words(composite_words_values, format="xml", disambiguate=disambiguate)
    else:
        composite_words_output = analyzer.analyze_words(composite_words_values, format="xml")
    composite_words_output_str = print_to_str(composite_words_output)

    offset = 0
    for key in composite_words.keys():
        search_pattern = '<w><ana lex="" gr="" parts="" gloss=""></ana>' + composite_words[key] + '</w>'
        if composite_words_output_str.find(search_pattern) != -1:
            parts = composite_words[key].split("-")
            wordlist = wordlist[0:key+offset] + parts + wordlist[key+1+offset:]
            offset += len(parts) - 1

    if has_disamb:
        parser_output = analyzer.analyze_words(wordlist, format="xml", disambiguate=disambiguate)
    else:
        parser_output = analyzer.analyze_words(wordlist, format="xml")
    parser_output_str = print_to_str(parser_output)

    return insert_parser_output_to_text(dedoc_output, parser_output_str, lang=lang)

def apertium_parser(dedoc_output, apertium_path, lang):

    def reformat(biltrans_filename="", morph_filename="", multi_filename="", bilingual=False, multi=False):

        skip_list = ["guio", "cm", "sent", "lpar", "rpar", "lquot", "rquot"]

        def to_skip(element):
            for skip_elem in skip_list:
                if element.find(skip_elem) != -1:
                    return True
            return False

        def gr(elem):
            gr_list = list()
            for match in re.findall(r"<(\w+)>", elem.split("/")[0]):
                gr_list.append(match)
            return gr_list

        def add_gr(gr_list):
            to_add = " gr=\""
            if gr_list:
                k = 0
                N = len(gr_list)
                for gr_elem in gr_list:
                    if k == 0:
                        to_add += gr_elem.upper()
                        if N > 1:
                            to_add += ","
                    elif k < N - 1:
                        to_add += gr_elem + ","
                    else:
                        to_add += gr_elem
                    k += 1
            to_add += "\""
            return to_add

        def add_variant(lex, gr, trans, parts = " parts=\"\"", gloss = " gloss=\"\""):
            return "<ana " + lex + gr + parts + gloss + trans + " ></ana>"

        parsed = ""

        morph_file = open(morph_filename, "r", encoding="UTF-8")
        morph = morph_file.read()
        morph_elements = re.findall(r"\^(.+?)\$", morph)

        if not bilingual:

            if multi:
                multi_file = open(multi_filename, "r", encoding="UTF-8")
                multi = multi_file.read()
                multi_elements = re.findall(r"\^(.+?)\$", multi)

            i = -1
            for morph_element in morph_elements:

                i += 1
                if len(morph_element) == 0 or to_skip(morph_element):
                    continue

                new = ""
                new += "<w>"

                if morph_element.find("*") != -1:
                    new += add_variant(" lex=\"\"", " gr=\"\"", " trans_ru=\"\"")
                    continue

                morph_slash_split_list = morph_element.split("/")
                orig = morph_slash_split_list[0]

                for lex_and_gr in morph_slash_split_list[1:]:
                    new_lex = "lex=" + "\"" + lex_and_gr[:lex_and_gr.find("<")].lower() + "\""
                    if not multi:
                        new_gr = add_gr(gr(lex_and_gr))
                    else:
                        new_gr = add_gr(gr(multi_elements[i]))
                    new_variant = add_variant(new_lex, new_gr, " trans_ru=\"\"")
                    if new.find(new_variant) == -1:
                        new += new_variant
                    if multi:
                        break

                new += orig
                new += "</w>"
                new += "\n"

                parsed += new

            return parsed

        biltrans_file = open(biltrans_filename, "r", encoding="UTF-8")
        biltrans = biltrans_file.read()
        biltrans_elements = re.findall(r"\^(.+?)\$", biltrans)

        def is_conform(lex_1, lex_2):
            if lex_1.lower().find(lex_2.lower()) == -1 and lex_2.lower().find(lex_1.lower()) == -1:
                return False
            else:
                return True

        def biltrans_elem(i):
            return (
                biltrans_elements[i] if i < len(biltrans_elements) else ""
            )

        def biltrans_lex(i):
            return (
                biltrans_elem(i)[0: biltrans_elem(i).find("<")].lower()
            )

        def trans(elem):
            trans_list = list()
            for match in re.findall(r"/(\w+)<", biltrans_elem(i)):
                if match not in trans_list:
                    trans_list.append(match)
            return trans_list

        i = -1
        for morph_element in morph_elements:
            i += 1

            if len(morph_element) == 0:
                continue

            new = ""
            new += "<w>"

            if morph_element.find("*") != -1:
                new += add_variant(" lex=\"\"", " gr=\"\"", " trans_ru=\"\"")
                continue

            if to_skip(morph_element):
                continue

            morph_slash_split_list = morph_element.split("/")
            orig = morph_slash_split_list[0]
            lex = morph_slash_split_list[1][:morph_slash_split_list[1].find("<")]

            offset = 0
            match_found = True
            while not is_conform(lex, biltrans_lex(i+offset)):
                if offset > 2:
                    match_found = False
                    break
                offset += 1
            if not match_found:
                continue
            i += offset

            new_lex = "lex=" + "\"" + biltrans_lex(i).lower() + "\""
            new_gr = add_gr(gr(biltrans_elem(i)))
            trans_list = trans(biltrans_elem(i))

            if not trans_list and new_gr == " gr=\"\"":
                new += add_variant(" lex=\"\"", " gr=\"\"", " trans_ru=\"\"")
                continue

            if trans_list:
                for trans_elem in trans_list:
                    new += add_variant(new_lex, new_gr, " trans_ru=\"" + trans_elem.lower() + "\"")
            else:
                new += add_variant(new_lex, new_gr, " trans_ru=\"\"")

            new += orig
            new += "</w>"
            new += "\n"

            parsed += new

        biltrans_file.close()
        morph_file.close()

        return parsed

    if lang in ['kaz', 'tat', 'bak-tat']:
        bilingual = True
    else:
        bilingual = False

    if lang in ['sah']:
        multi = True
    else:
        multi = False

    if bilingual:
        biltrans_file_id, biltrans_filename = tempfile.mkstemp()

    if multi:
        multi_file_id, multi_filename = tempfile.mkstemp()

    morph_file_id, morph_filename = tempfile.mkstemp()

    dedoc_output_without_tags = re.sub(r"(<.*?>)|&nbsp", "", dedoc_output)

    input_file_id, input_filename = (
        tempfile.mkstemp(text = True))

    with open(input_filename, 'w') as input_file:
        input_file.write(dedoc_output_without_tags)

    s1 = 0
    s2 = 0
    if lang == 'tat':
        s1 = os.system("cat \"" + input_filename + "\" | apertium -d " + apertium_path + "/apertium-tat-rus tat-rus-biltrans >> " +  biltrans_filename)
        s2 = os.system("cat \"" + input_filename + "\" | apertium -d " + apertium_path + "/apertium-tat-rus tat-rus-morph >> " +  morph_filename)

    elif lang == 'kaz':
        s1 = os.system("cat \"" + input_filename + "\" | apertium -d " + apertium_path + "/apertium-kaz-rus kaz-rus-biltrans >> " + biltrans_filename)
        s2 = os.system("cat \"" + input_filename + "\" | apertium -d " + apertium_path + "/apertium-kaz-rus kaz-rus-morph >> " + morph_filename)

    elif lang == 'bak-tat':
        s1 = os.system("cat \"" + input_filename + "\" | apertium -d " + apertium_path + "/apertium-tat-bak bak-tat-biltrans >> " + biltrans_filename)
        s2 = os.system("cat \"" + input_filename + "\" | apertium -d " + apertium_path + "/apertium-tat-bak bak-tat-morph >> " + morph_filename)

    elif lang == 'sah':
        s1 = os.system("cat \"" + input_filename + "\" | apertium -d " + apertium_path + "/apertium-sah sah-morph >> " + morph_filename)
        s2 = os.system("cat \"" + input_filename + "\" | apertium -d " + apertium_path + "/apertium-sah sah-multi >> " + multi_filename)

    else:
        s1 = os.system(f"cat '{input_filename}' | apertium -d {apertium_path}/apertium-{lang} {lang}-morph >> {morph_filename}")
        s2 = s1

    if s1 != 0 or s2 != 0:
        raise ValueError("An error occured during Apertium parser process running")

    if bilingual:
        parser_output = reformat(biltrans_filename=biltrans_filename, morph_filename=morph_filename, bilingual=True)
        os.remove(biltrans_filename)
    elif multi:
        parser_output = reformat(multi_filename=multi_filename, morph_filename=morph_filename, multi=True)
        os.remove(multi_filename)
    else:
        parser_output = reformat(morph_filename=morph_filename)

    os.remove(morph_filename)
    os.remove(input_filename)

    return insert_parser_output_to_text(dedoc_output, parser_output, lang=lang)

def hfst_parser(dedoc_output, lang, debug_flag=False):

    if debug_flag:
        with open("dedoc_output", 'w') as f:
            print(dedoc_output, file=f)

    parser_path = f"/opt/hfst/{lang}"

    with open(f"{parser_path}/lexicon.lexc", 'r') as f:
        lexicon = f.read()

    xfst = HfstTransducer.read_from_file(f"{parser_path}/rules.xfst.hfst")
    xfst.invert()

    sent_regex = re.compile(r'[.|!|?|...]')
    word_regex = re.compile(r'[,| |:|"|-|*]')

    words = 0
    analyzed = 0
    parser_list = []

    # remove html tags from dedoc_output
    text = fromstring(dedoc_output).text_content()
    sentences = filter(lambda t: t, [t.strip() for t in sent_regex.split(text)])
    for s in sentences:
        wordlist = filter(lambda t: t, [t.strip() for t in word_regex.split(s)])
        for w in wordlist:
            words = words + 1
            lookup = xfst.lookup(w)
            if len(lookup) == 0:
                lookup = xfst.lookup(w.lower())
            if len(lookup) > 0:
                analyzed = analyzed + 1
                section = "'<w>"
                for lkp in map(lambda l: l[0], lookup):

                    if '+' in lkp:
                        plus_pos = lkp.index('+')
                        lex = lkp[:plus_pos]
                        gr = lkp[plus_pos + 1:].replace('+', ',')
                    else:
                        lex = lkp
                        gr = "Unknown"

                    # Get translation
                    if ((xln := re.search(f"[\r\n]{lex}:{w} .*!([^0].*)[\r\n]", lexicon)) or
                        (xln := re.search(f"[\r\n]{lex}:{w.lower()} .*!([^0].*)[\r\n]", lexicon)) or
                        (xln := re.search(f"[\r\n]{lex}:.*!([^0].*)[\r\n]", lexicon))):
                        xln = xln.group(1)
                    else:
                        xln = "Unknown"

                    section += f'<ana lex={lex} gr={gr} parts="" gloss="" trans_ru={xln}></ana>'
                section += f"{w}</w>'"
                parser_list.append(section)
            else:
                parser_list.append(f'\'<w><ana lex="" gr="" parts="" gloss=""></ana>{w}</w>\'')

    parser_output = ", ".join(parser_list)

    if debug_flag:
        with open("parser_output", 'w') as f:
            print(parser_output, file=f)
        print(f"Analyzed per word: {analyzed / words}")

    return insert_parser_output_to_text(dedoc_output, parser_output, lang=lang)

def timarkh_udm(dedoc_output):
    return timarkh_uniparser(dedoc_output, 'udm')

def timarkh_erzya(dedoc_output):
    return timarkh_uniparser(dedoc_output, 'erzya')

def timarkh_moksha(dedoc_output):
    return timarkh_uniparser(dedoc_output, 'moksha')

def timarkh_komi_zyryan(dedoc_output):
    return timarkh_uniparser(dedoc_output, 'komi_zyryan')

def timarkh_meadow_mari(dedoc_output):
    return timarkh_uniparser(dedoc_output, 'meadow_mari')

def apertium_tat_rus(dedoc_output, apertium_path):
    return apertium_parser(dedoc_output, apertium_path, 'tat')

def apertium_kaz_rus(dedoc_output, apertium_path):
    return apertium_parser(dedoc_output, apertium_path, 'kaz')

def apertium_sah(dedoc_output, apertium_path):
    return apertium_parser(dedoc_output, apertium_path, 'sah')

def apertium_bak_tat(dedoc_output, apertium_path):
    return apertium_parser(dedoc_output, apertium_path, 'bak-tat')

def apertium_bak(dedoc_output, apertium_path):
    return apertium_parser(dedoc_output, apertium_path, 'bak')

def apertium_rus(dedoc_output, apertium_path):
    return apertium_parser(dedoc_output, apertium_path, 'rus')

def apertium_eng(dedoc_output, apertium_path):
    return apertium_parser(dedoc_output, apertium_path, 'eng')

def apertium_tuv(dedoc_output, apertium_path):
    return apertium_parser(dedoc_output, apertium_path, 'tyv')

def apertium_hye(dedoc_output, apertium_path):
    return apertium_parser(dedoc_output, apertium_path, 'hye')

def apertium_krc(dedoc_output, apertium_path):
    return apertium_parser(dedoc_output, apertium_path, 'krc')

def hfst_kalmyk(dedoc_output):
    return hfst_parser(dedoc_output, 'xal')

def hfst_ancient_kalmyk(dedoc_output):
    return hfst_parser(dedoc_output, 'anc_xal')
