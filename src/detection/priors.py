"""Prior probability that an object of a given class is moving (option A: hand list + default).

LaValle, *Virtual Reality* Sec. 6.4: perception starts from a prior P(h) built from
experience ("people move, walls don't") and updates it with cues. This table is
that prior. Step 8 of the pipeline (motion check) will update it per object with
depth / edge / self-motion cues; for now it only decides which detections are
"suspects" (P >= SUSPECT_THRESHOLD) and must be kept out of the first pose estimate.

Rules:
  * every name not listed gets DEFAULT_PRIOR - small but never 0, because a prior
    of 0 can never be raised by evidence (Bayes: 0 x anything = 0)
  * names are matched exactly (lower case); the lists cover the COCO-80 names,
    the YOLOE text-prompt list, and the person / clothing / carried-object words
    of the YOLOE prompt-free vocabulary (4,585 names, read through by hand). The
    prompt-free model rarely says "person": it says "manager", "researcher",
    "polo shirt", "shorts" - so occupations, clothing and body parts are listed.
"""
from __future__ import annotations

DEFAULT_PRIOR = 0.01
SUSPECT_THRESHOLD = 0.3

PERSON = 0.8          # people, groups of people, human actions
WORN = 0.7            # clothing, accessories, body parts: move with a person
ANIMAL = 0.7
FLYING_TOY = 0.6      # balloons, balls: pushed / thrown around
VEHICLE = 0.5
CARRIED = 0.4         # boxes, bags, luggage: what people pick up and carry
MOVABLE_FURNITURE = 0.3   # chairs, carts, doors: moved now and then
HANDHELD = 0.1        # cups, phones, books: move, but rarely and are small

_GROUPS: list[tuple[float, str, str]] = [
    (PERSON, "person", """
person | people | man | woman | boy | girl | child | baby | toddler | adult | teenager | youth |
newborn | senior | individual | dude | crowd | group | family | couple | twin | sibling | cousin |
daughter | father | mother | grandfather | grandmother | grandparent | neighbor | partner | friend |
actor | airman | animator | architect | artist | astronaut | astronomer | athlete | auto mechanic |
award winner | baker | ballerina | ballet dancer | barber | barista | bartender | baseball player |
baseball pitcher | baseball team | basketball coach | basketball player | basketball team | bassist |
batman | batter | beekeeper | biker | bishop | blacksmith | blue artist | bodybuilder | boxer | bride |
groom | bridesmaid | brunette | builder | bus driver | business executive | business team |
business woman | businessman | butcher | camper | captain | carpenter | catcher | celebrity |
champion | chef | chemist | child actor | choir | client | climber | coach | college student |
commander | commentator | commuter | competitor | composer | conductor | construction worker | cook |
costumer film designer | country artist | country pop artist | cowboy | craftsman | crew |
cricketer | cricket team | customer | dancer | defender | dentist | designer | detective | director |
disciple | diver | dj | doctor | driver | drummer | duchess | educator | electrician | engineer |
entertainer | explorer | farmer | fashion designer | fashion girl | fashion model | figure skater |
film director | film producer | fireman | fisherman | flight attendant | florist | flower girl |
folk artist | folk rock artist | football coach | football player | football team | gang | gardener |
geisha | gladiator | goalkeeper | golfer | graduate | guard | guide | guitarist | gymnast | hacker |
hard rock artist | head coach | hero | hiker | hip hop artist | historian | hockey player | host |
ice hockey player | ice hockey team | infantry | jazz artist | jazz fusion artist | jockey |
journalist | judge | king | lawyer | lead singer | leader | lifeguard | lumberjack | magician |
mahout | maid | makeup artist | manager | marching band | martial artist | matador | mechanic |
medical staff | metal artist | miner | minister | monk | motorcycle racer | motorcyclist |
mountaineer | mountain biker | music video performer | musician | nanny | newlywed | nun | nurse |
officer | official | painter | paramedic | paratrooper | participant | passenger | patient |
pedestrian | philosopher | photographer | physicist | pianist | pilgrim | pilot | pirate | player |
plumber | police | politician | pop artist | pope | preacher | president | prince | princess |
professional | professor | prophet | protester | queen | rapper | record producer | referee |
rescuer | researcher | rider | rock artist | rock band | rock climber | rocker | rugby player |
runner | sailor | samurai | santa claus | saxophonist | scientist | scout | sculptor | secretary |
security | shepherd | shopper | singer | skateboarder | skater | skier | snowboarder |
soccer goalkeeper | socialite | soldier | spectator | sport team | squad | staff | street artist |
street vendor | striker | student | supermodel | superhero | supporter | surfer | surgeon | swimmer |
tailor | tattoo artist | teacher | technician | tennis player | theatre actor | tour guide | tourist |
trainer | tribe | trio | troop | truck driver | tv actor | tv personality | vendor | veteran | vet |
violinist | violist | volleyball player | volunteer | waiter | warrior | wedding couple |
wedding party | wedding photographer | welder | witch | worker | wrestler | writer | zombie | joker |
cosplay | shirtless | selfie | team photo | group photo | family photo | couple photo |
sit | sitting | stand | standing | walk | walking | run | jog | dance | hug | kiss | handshake |
kneel | squat | stretch | talk | conversation | interview | carry | pose | lean | smile | laugh |
wear | stroll | huddle | greet | applause
"""),
    (WORN, "worn/body", """
afro | ankle | arm | back | beard | belly | bangs | braid | chest | curl | ear | eye | eyebrow |
face | face close-up | foot | forehead | freckle | hair | hair color | haircut | hairstyle | hand |
head | heel | hip | jaw | lap | leg | limb | long hair | mouth | muscle | neck | nose | ponytail |
pigtail | shoulder | short hair | skin | stomach | tattoo | toe | waist | wrinkle | profile | scar |
accessory | apron | armband | armor | anklet | aviator sunglasses | baby clothe | badge | bandeau |
baseball hat | baseball uniform | bathrobe | beret | belt | belt buckle | bib | bikini |
bikini top | blouse | bolo tie | bonnet | boot | bow tie | bracelet | brooch | broach |
bulletproof vest | business suit | cap | cape | cardigan | chain mail | cheongsam | christmas hat |
cloak | clogs | clothe | clothing | coat | cocktail dress | corset | costume | coverall |
cowboy boot | cowboy hat | crop top | cuff | cufflink | denim jacket | dress | dress hat |
dress shirt | dress shoe | dress suit | earmuff | earphone | earring | evening dress | eye mask |
fashion accessory | fashion look | fedora | flannel | flip-flop | football helmet | footwear |
fur coat | garment | gasmask | glasses | glove | goggles | golf cap | gown | halloween costume |
halter top | hanfu | hat | headdress | headscarf | helmet | high heel | hiking boot | hood |
hoodie | jacket | jeans | jewellery | jockey cap | jumpsuit | khaki | kilt | kimono | knee pad |
lab coat | lace dress | lanyard | leather jacket | leather shoe | legging | leotard | life jacket |
loafer | maxi dress | military uniform | miniskirt | mask | motorcycle helmet | neckband |
necklace | neckline | overall | overcoat | paisley bandanna | pant | pants | pantyhose | party hat |
pendant | plug hat | polo neck | polo shirt | poncho | protective suit | rain boot | raincoat |
robe | roller skate | roller skates | running shoe | safety helmet | safety vest | sandal | sari |
scarf | school uniform | shawl | shirt | shoe | shorts | shower cap | silk stocking | ski boot |
ski jacket | skiing shoes | skullcap | sleepwear | sleeve | slipper | sock | sombrero |
sports coat | sportswear | stocking | straw hat | suit | sun hat | sunglasses | suspenders |
sweat pant | sweatband | sweater | sweatshirt | swim cap | swimwear | t shirt | t-shirt | tiara |
tie | tights | trench coat | underclothes | underdrawers | uniform | veil | vest | visor |
waistband | wedding dress | wedding ring | wet suit | wetsuit | wig | witch hat | wrap dress |
wristband | watch | camouflage
"""),
    (ANIMAL, "animal", """
animal | pet | mammal | dog | puppy | cat | kitten | horse | cow | sheep | goat | pig | rabbit |
bunny | hamster | guinea pig | rat | squirrel | bird | parrot | pigeon | chicken | duck | goose |
monkey | ape | bear | elephant | zebra | giraffe | deer | fox | wolf | lion | tiger | beagle |
bulldog | labrador | golden retriever | german shepherd | poodle | pug | husky | chihuahua | corgi |
dachshund | dalmatian | doberman | greyhound | pitbull | rottweiler | samoyed | schnauzer |
shiba inu | pomeranian | retriever | sheepdog | street dog | police dog | guard dog | tabby |
siamese | persian cat | feline | kangaroo | koala | panda | penguin | insect | butterfly | fly |
bee | mosquito
"""),
    (FLYING_TOY, "flying/toy", """
balloon | balloon arch | ball | beach ball | sports ball | football | basketball | volleyball |
rugby ball | frisbee | kite | soap bubble | drone | hula hoop
"""),
    (VEHICLE, "vehicle", """
car | bus | truck | lorry | bicycle | motorcycle | motorbike | scooter | moped | vespa | van |
taxi | train | segway | hoverboard | skateboard | tricycle | monocycle | golf cart | forklift |
tractor | robot | wheelchair | baby carriage | mobility scooter | airplane | boat
"""),
    (CARRIED, "carried", """
bag | backpack | box | cardboard box | carton | crate | storage box | gift box | shoe box |
package | luggage | suitcase | briefcase | handbag | shoulder bag | tote bag | messenger bag |
duffel | sack | shopping bag | paper bag | grocery bag | satchel | hatbox | cigar box | toolbox |
lunch box | umbrella | basket | laundry basket | shopping basket | gift bag | mountaineering bag
"""),
    (MOVABLE_FURNITURE, "movable furniture", """
chair | office chair | computer chair | swivel chair | folding chair | bar stool | stool |
rocking chair | feeding chair | beach chair | bean bag chair | armchair | trolley | cart |
handcart | shopping cart | dolly | door | car door | glass door | screen door | barn door |
bathroom door | elevator door | garage door
"""),
    (HANDHELD, "handheld", """
phone | cell phone | smartphone | iphone | cup | mug | coffee cup | paper cup | bottle |
glass bottle | beer bottle | can | book | notebook | tablet | ipad | tablet computer | remote |
game controller | joystick | camera | camcorder | magazine | wine glass | teddy bear | doll |
stuffed animal | stuffed toy
"""),
]


def _parse(block: str) -> list[str]:
    return [n.strip().lower() for n in block.replace("\n", " ").split("|") if n.strip()]


PRIORS: dict[str, float] = {}
GROUP_OF: dict[str, str] = {}
for _p, _group, _block in _GROUPS:
    for _name in _parse(_block):
        if _name not in PRIORS:        # first (highest) group wins if a name is listed twice
            PRIORS[_name], GROUP_OF[_name] = _p, _group


def prior_for(class_name: str) -> float:
    """P(moving) before any motion evidence, for a detector class name."""
    return PRIORS.get(class_name.strip().lower(), DEFAULT_PRIOR)


def is_suspect(prior: float) -> bool:
    return prior >= SUSPECT_THRESHOLD


def group_of(class_name: str) -> str:
    return GROUP_OF.get(class_name.strip().lower(), "default")
