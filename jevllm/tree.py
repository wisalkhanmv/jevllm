"""A categorised vocabulary, and the tree walk that reaches into it.

A Choice accepts at most 255 options, which caps a flat vocabulary hard: the
`common` strategy can offer 246 words and the rest of the corpus is simply
unreachable on that step. A tree removes the ceiling. Ask which *category* the
next word belongs to, then which word within that category, and the reachable
vocabulary becomes categories x 255 rather than 255.

The descent costs no extra latency. Jev ingests the state once and evaluates
every question in a request against it in parallel, so the category question
and one word question per category all travel together and only the branch
that wins is read. One round trip, whole tree.

It also suits the model better than a flat list does. "Is the next word a
person, an action, or a connective?" is a judgment between options that mean
genuinely different things — which is what Jev is calibrated for — where
picking one of 246 flat words is not.
"""

from __future__ import annotations

#: Each branch: (description shown to the model, words).
#: Descriptions matter — they are what Jev judges between at the first level,
#: so they say what the category *is for*, not merely what it is called.
TREE: dict[str, tuple[str, list[str]]] = {
    "determiner": (
        "an article or determiner introducing a noun: the, a, this, some",
        """the a an this that these those some any all both each every no
        another other such my your his her its our their""".split(),
    ),
    "pronoun": (
        "a pronoun standing in for a person or thing: I, you, it, they",
        """i you he she it we they me him her us them myself yourself himself
        herself itself ourselves themselves who whom whose which what someone
        anyone everyone nobody something anything everything nothing one""".split(),
    ),
    "auxiliary": (
        "a helping or linking verb: is, was, have, will, can, would",
        """is are was were be been being am has have had having do does did
        will would shall should may might must can could need used going""".split(),
    ),
    "verb": (
        "a main verb naming an action or process: go, make, think, said",
        """go goes going went gone come comes coming came take takes took taken
        make makes making made say says said tell tells told know knows knew
        think thinks thought see sees saw seen look looks looking looked
        want wants wanted give gives gave find finds found use uses used
        work works worked call calls called try tries tried ask asks asked
        need needs feel feels felt become leave leaves left put puts keep kept
        begin began seem seems help helps talk talks turn turns start started
        show shows heard play plays run runs move moves like likes liked
        live lives lived believe hold holds bring brought happen happened
        write wrote sit sits stand stood lose lost pay paid meet met
        learn learned change changed lead led understand watch follow stop
        speak spoke read allow add spend grow open opens walk walks win won
        offer remember love loves consider appear buy bought wait waits
        send sent expect build built stay stayed fall fell cut reach remain
        eat eats ate drink drinks sleep sleeps wake drive drove sing sang
        enjoy enjoys enjoyed prefer choose chose pick picked bring brings""".split(),
    ),
    "noun_person": (
        "a noun naming a person, group or role: friend, mother, team, people",
        """person people man woman child children friend friends family father
        mother brother sister parent parents team group everyone someone
        neighbour colleague student teacher doctor stranger crowd guest
        customer partner boss writer artist reader driver visitor""".split(),
    ),
    "noun_place": (
        "a noun naming a place or location: home, city, street, country",
        """home house room door window city town country world place street road
        garden kitchen office school park station airport shop market beach
        river mountain island village building floor corner side area""".split(),
    ),
    "noun_thing": (
        "a noun naming a concrete object or substance: car, book, water, phone",
        """car book books phone email letter message note paper bag box key
        table chair bed light water food bread coffee tea rice pizza fruit
        money card picture photo film music game clock window glass door
        computer screen machine engine clothes shoes hand head eye face""".split(),
    ),
    "noun_time": (
        "a noun naming a time, period or occasion: day, morning, week, year",
        """time day days week weeks month months year years hour hours minute
        minutes second moment morning afternoon evening night tonight today
        tomorrow yesterday weekend summer winter spring autumn season future
        past present age period while dawn noon midnight birthday holiday""".split(),
    ),
    "noun_idea": (
        "a noun naming something abstract: reason, problem, life, question",
        """thing things part end way reason problem question answer idea fact
        case point life work job business company school story name word
        number line order change chance choice kind sort sense truth
        power right mind heart health hope fear love luck plan result""".split(),
    ),
    "adjective": (
        "a describing word qualifying a noun: good, cold, large, difficult",
        """good bad better best worse worst new old young high low long short
        big small large little happy sad glad sure certain clear easy hard
        difficult simple strong weak early late right wrong true false real
        whole full empty free busy tired ready warm cold hot cool bright dark
        heavy light quick slow quiet loud safe close open deep fine great
        nice lovely beautiful terrible awful amazing interesting boring
        serious important different same own next last other favourite""".split(),
    ),
    "adverb": (
        "a word modifying a verb or sentence: quickly, never, quite, here",
        """now then here there today always never often sometimes usually rarely
        again soon later already almost nearly hardly quite rather enough very
        too also just only even still yet more most much less least together
        anyway instead however therefore perhaps maybe probably certainly
        really actually honestly obviously exactly absolutely definitely
        quickly slowly carefully suddenly finally simply clearly well""".split(),
    ),
    "preposition": (
        "a word placing something in relation to another: in, with, after, from",
        """of in on at to from by with for about into over under after before
        between through during without within against across behind beyond
        near among around along beside despite toward up down out off past
        like as than since until per via inside outside upon""".split(),
    ),
    "conjunction": (
        "a word joining clauses: and, but, because, although, while",
        """and or but so because if then than that although though unless
        whether while when where why how since until once whereas plus
        nor yet either neither both""".split(),
    ),
    "reply": (
        "an opener, closer or discourse marker used in conversation: yes, thanks, well",
        """yes no yeah okay ok hi hello hey bye goodbye thanks thank please
        sorry sure right well fine good great nice cheers welcome indeed
        course alright hmm oh ah wow congratulations excuse pardon""".split(),
    ),
    "number": (
        "a number or quantity word: one, three, many, several, first",
        """one two three four five six seven eight nine ten eleven twelve
        twenty thirty fifty hundred thousand million first second third
        many few several dozen half couple single double none""".split(),
    ),
    "punctuation": (
        "a mark ending or dividing the sentence: full stop, comma, question mark",
        [".", ",", "?", "!"],
    ),
}

CATEGORY_INSTRUCTIONS = (
    "The text below is an excerpt from a longer document, cut off mid-sentence. "
    "What kind of word comes next, immediately after the last word shown? "
    "Judge by grammar and meaning: what would a fluent writer put here?"
)

WORD_INSTRUCTIONS = (
    "The text below is an excerpt from a longer document, cut off mid-sentence. "
    "Assume the next word is {desc}. Which one of these continues the text best?"
)

#: Options per word question. Categories are well under this; the cap is a
#: guard rather than a limit anyone is expected to hit.
MAX_PER_CATEGORY = 255


def categories() -> dict[str, str]:
    """The first-level Choice: category name -> what it is for."""
    return {name: desc for name, (desc, _) in TREE.items()}


def words(name: str) -> list[str]:
    _, ws = TREE[name]
    return list(dict.fromkeys(ws))[:MAX_PER_CATEGORY]


def describe(name: str) -> str:
    return TREE[name][0]


def size() -> tuple[int, int]:
    """(categories, unique words) — what the tree can actually reach."""
    everything = {w for _, ws in TREE.values() for w in ws}
    return len(TREE), len(everything)
