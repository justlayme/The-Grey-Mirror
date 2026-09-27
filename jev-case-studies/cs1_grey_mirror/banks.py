"""Message banks for the synthetic relationship threads.

Every bank is plain text written for this benchmark. `{slot}` fields are filled at generation time.
Banks come in three registers for the partner ("B"):
  WARM         engaged, curious, affectionate
  COLD_SUBTLE  polite, low-engagement, deflecting or passive-aggressive. Written so the *wording*
               stays neutral-to-positive: the change is in meaning, not in sentiment vocabulary.
  COLD_OVERT   explicitly negative, hostile or dismissive (the easy case a lexicon should catch)
"""

SLOTS = {
    "pet": ["babe", "love", "hon", "baby", "sweetheart", "cutie"],
    "food": ["tacos", "pho", "pizza", "ramen", "sushi", "thai food", "burgers", "pasta", "curry"],
    "act": ["a walk", "a movie", "the farmers market", "a hike", "mini golf", "bowling",
            "the bookstore", "trivia night", "the museum", "karaoke"],
    "place": ["the lake", "your mom's", "downtown", "the beach", "the mountains", "the coast",
              "the park", "the new brewery"],
    "work": ["the meeting", "my presentation", "the deadline", "inventory", "the client call",
             "training", "the audit", "the quarterly report"],
    "cw": ["Dana", "Marcus", "Priya", "Tom", "Keisha", "Luis", "Nora", "Omar"],
    "day": ["saturday", "sunday", "friday night", "next weekend", "thursday"],
    "time": ["6", "7", "7:30", "8", "noon"],
}

NAME_PAIRS = [
    ("Sam", "Alex"), ("Jordan", "Riley"), ("Casey", "Morgan"), ("Taylor", "Jamie"),
    ("Avery", "Quinn"), ("Drew", "Parker"), ("Rowan", "Skyler"), ("Emerson", "Reese"),
    ("Hayden", "Logan"), ("Blake", "Cameron"),
]

# --- check-in ------------------------------------------------------------------------------------
CHECKIN_Q = [
    "how's your day going?", "how was {work}?", "hey you, how's work treating you today",
    "did {work} go ok?", "how are you feeling today?", "morning! sleep ok?",
    "how'd it go with {cw}?", "you surviving today?", "what are you up to rn", "hows the day been",
    "hey {pet}, how's your afternoon", "did you end up going to the gym?",
]
CHECKIN_WARM = [
    "honestly so good, {work} went better than i expected!! how's yours going?",
    "long but good! {cw} brought donuts so i'm happy lol. how about you?",
    "better now that you texted 😊 how's your day been?",
    "pretty great actually, i'll tell you all about it tonight. did you eat yet?",
    "busy but fine, i keep thinking about {act} this weekend haha. what are you up to?",
    "so good! i finally finished {work}, i feel like a new person. how are you?",
    "it's been a lot but i'm ok, thanks for checking on me ❤️ how's yours?",
    "not bad at all! {cw} was being hilarious today, remind me to tell you. hows work?",
    "slept great! you were right about the tea lol. how'd you sleep?",
    "kind of hectic but talking to you helps. what's new with you?",
    "good!! got out early so i'm making {food} tonight, want some?",
    "so tired but in a good mood, can't wait to see you later. how's your day?",
]
CHECKIN_COLD_SUBTLE = [
    "it was good, same as usual. just a lot of emails and {work} stuff. did the bill get paid?",
    "good, thanks. busy with {work} so i'll be on my phone less today. is the car free tomorrow?",
    "fine thanks. {cw} asked me to cover the afternoon so i'll be late. can you feed the cat?",
    "yeah it's good, nothing really to report. did you see the email from the landlord?",
    "sure, it was okay. i'll grab something on the way home. do we need anything from the store?",
    "it's fine, thanks for asking. i have a lot going on today. what time is your appointment?",
    "all good. just busy. {work} took most of the morning, is the car free tomorrow?",
    "pretty normal day, nothing special. did the package come? i'll probably go to bed early.",
    "good thanks 🙂 it's been a long one. is it okay if i go to {place} after work by myself?",
    "it's okay. i have {work} again tomorrow so i'm trying to get ahead. did you call the plumber?",
    "fine, great actually. did you remember the internet bill? it's due friday.",
    "yep, good. {cw} and i are grabbing dinner after work, is that fine? don't wait up.",
]
CHECKIN_COLD_OVERT = [
    "honestly terrible and i don't want to talk about it",
    "why do you always ask me that, it's annoying",
    "bad. and i really don't need you checking on me every hour",
    "it sucks. everything sucks. leave it",
    "ugh i hate this job and i hate how everyone keeps bothering me today",
    "awful, and you texting me nonstop isn't helping",
    "not great. can you just give me some space today",
    "horrible. i'm angry at everyone rn including you tbh",
    "stop asking. it was bad.",
    "worst day. i'm so done with all of this",
    "i'm fine, stop hovering, it's exhausting",
    "miserable. i don't want to deal with anyone tonight",
]

A_AFTER_WARM = [
    "yay that makes me so happy!! 😊", "haha i love that. can't wait to hear about it", "good! mine's been ok, better now",
    "aww good. mine was fine, just busy", "love that for you!!", "glad it's a good one ❤️", "mine's been fine! see you tonight?",
    "haha nice. see you later 😊",
]
A_AFTER_NEUTRAL = ["ok", "got it", "alright", "sounds good", "ok cool", "k, thanks", "okay", "sure"]
A_AFTER_HURT = [
    "oh ok", "ok... you alright?", "did i do something?", "ok, text me when you can", "oh. ok 🙁",
    "you seem off, everything ok?", "hope it gets better", "okay 😕", "sorry, i'll leave you alone",
]

# --- plans ---------------------------------------------------------------------------------------
PLAN_PROPOSE = [
    "want to do {act} {day}?", "we should try {food} {day}, i've been craving it",
    "are you free {day}? thinking {act}", "what if we went to {place} {day}? just us",
    "dinner at {time} tonight?", "my sister invited us over {day}, want to go?",
    "let's plan something fun for {day} 😊", "movie night tonight? i'll pick up {food}",
    "come over after work?", "should we do {act} {day}? my treat",
]
PLAN_WARM = [
    "yes!! i've been wanting to do that forever 😍", "omg yes, count me in. i'll bring snacks",
    "that sounds perfect, i can't wait ❤️", "yesss {food} is exactly what i need. {time}?",
    "absolutely, it's a date 😘", "yes please, i miss just hanging out with you",
    "love that idea!! should we invite anyone or just us?", "i'm so in. let's make a whole day of it",
    "yes! i'll drive this time. so excited", "sounds amazing, i'll clear my schedule for you",
]
PLAN_COLD_SUBTLE = [
    "maybe, sounds fun. let me see how the week goes.", "sounds nice! not sure yet though, might have work stuff.",
    "sure, possibly. {cw} mentioned something that day too.", "we could, sure. i'll let you know 🙂",
    "that could be fun. maybe another time?", "yeah maybe, sounds good. no promises though.",
    "sounds great in theory 🙂 let's play it by ear.", "sure, sounds good if nothing else comes up.",
    "maybe next weekend instead? this one's full, sorry.", "nice idea. go with your friends if you want though?",
]
PLAN_COLD_OVERT = [
    "no. i really don't want to", "why would i want to do that with you right now",
    "not happening, i'm not in the mood for you tbh", "ugh no. stop planning stuff for us",
    "absolutely not, i'm still mad about last time", "no thanks, i'd rather be alone",
    "i hate that place and you know it", "can you stop pushing this? the answer is no",
    "nope. don't wait up either", "i'm not going anywhere with you this weekend",
]

# --- affection -----------------------------------------------------------------------------------
AFFECTION_A = [
    "miss you ❤️", "love you", "thinking about you", "can't wait to see you tonight",
    "you're the best part of my day", "i love you so much {pet}", "counting down till i see you 😘",
    "just wanted to say i'm lucky to have you",
]
AFFECTION_WARM = [
    "miss you more ❤️❤️", "love you too {pet} 😘", "aww you're gonna make me cry. love you",
    "i'm the lucky one, seriously", "thinking about you too, all day actually", "can't wait!! hurry home",
    "i love you so much it's ridiculous", "stoppp you're too cute 🥰 love you",
]
AFFECTION_COLD_SUBTLE = [
    "ok 🙂 see you later tonight", "you too, thanks. talk later", "haha thanks, that's nice",
    "mhm, see you later then", "that's sweet, thanks", "same. talk later, busy day",
    "noted lol 🙂", "ok cool, have a good day", "thanks, you too i guess", "👍 sounds good",
]
AFFECTION_COLD_OVERT = [
    "please don't", "i can't deal with this right now", "why are you being so clingy",
    "stop. seriously", "i don't feel the same way lately", "that's annoying honestly", "not now",
    "ugh. ok",
]

# --- partner-initiated share ---------------------------------------------------------------------
SHARE_WARM = [
    "saw this dog at the park and thought of you 😂", "{cw} just said the funniest thing, i'll tell you tonight lol",
    "guess what!! i got the thing i applied for!!", "this song came on and now i miss you",
    "found a new {food} spot we HAVE to try", "remember {place}? just saw a pic from that trip ❤️",
    "i'm making your favorite for dinner 😊", "you would have loved this sunset",
]
SHARE_COLD_SUBTLE = [
    "fyi the landlord emailed about the lease, it's in your inbox. thanks!", "the car is making that noise again, i'll call the shop. no big deal.",
    "i'm going to {place} with {cw} on {day}, just so you know 🙂", "can you move your stuff off the counter? thanks.",
    "home late tonight, there's good leftover {food} in the fridge.", "your package came, i left it by the door 👍",
    "reminder the dentist moved your appointment to thursday. good luck!", "paid the electric bill, venmo me half whenever 🙂",
]
SHARE_COLD_OVERT = [
    "you left the kitchen a disaster again. seriously?", "i'm going out tonight and i don't want to hear about it",
    "your mom called again. deal with her yourself", "i'm so tired of cleaning up after you",
    "don't bother waiting up", "honestly this week has been awful because of you",
    "can you not be home when i get back", "i'm sleeping at my sister's tonight. don't call",
]
A_AFTER_SHARE_WARM = [
    "hahaha love it", "omg yes!!", "stop that's so cute", "aww 🥰", "lol tell me everything later", "ooh yes let's go",
    "that's amazing!!",
]

# --- neutral logistics (identical in every state) ---------------------------------------------------
LOGISTICS = [
    ("can you grab milk on the way home?", ["yep got it", "sure", "on it"]),
    ("did you feed the cat?", ["yes, this morning", "yep", "not yet, will do"]),
    ("what time is your appointment tomorrow?", ["3:30 i think", "at 10", "after lunch"]),
    ("is the car low on gas?", ["a little, i'll fill it up", "it's fine", "yeah kinda"]),
    ("don't forget trash day", ["on it", "already did it", "yep"]),
    ("do we need anything from the store?", ["just coffee and eggs", "paper towels", "nope, we're good"]),
    ("the wifi is down again", ["ugh ok i'll restart the router", "weird, mine works", "i'll call them"]),
    ("did you see the email from the landlord?", ["yeah i'll reply later", "not yet", "yep, it's fine"]),
    ("where did you put the spare key?", ["top drawer", "in the blue bowl", "on the hook"]),
    ("is the dishwasher clean or dirty?", ["clean", "dirty, didn't run it", "clean, just ran it"]),
]

# --- goodnight -----------------------------------------------------------------------------------
GOODNIGHT_A = ["night ❤️", "going to bed, love you", "sleep well {pet}", "night night 😴", "heading to sleep, see you tomorrow"]
GOODNIGHT_WARM = ["night {pet}, sweet dreams 😘", "love you, sleep well ❤️", "night!! can't wait for tomorrow",
                  "sleep tight, text me in the morning 🥰", "night babe, dream of me lol"]
GOODNIGHT_COLD_SUBTLE = ["ok good night, sleep well 🙂", "yep, night. have a good one", "sleep well, see you tomorrow",
                         "night, thanks. see you", "ok, good night 🙂 early day tomorrow"]
GOODNIGHT_COLD_OVERT = ["whatever. night", "k", "don't text me this late", "fine, go to bed", "ok bye"]

# --- lateness (where sarcasm shows up) ----------------------------------------------------------------
LATE_A = ["sorry, running like 20 min late 😬", "ugh traffic, gonna be late", "forgot to text back earlier sorry!",
          "i totally forgot to pick up {food}, sorry", "sorry, meeting ran over, leaving now"]
LATE_WARM = ["no worries at all! drive safe ❤️", "all good, take your time 😊", "haha it's fine, i'm not going anywhere",
             "no stress, we can grab it tomorrow", "all good! see you soon"]
LATE_COLD_SUBTLE = ["oh great, love that for me 🙂", "no worries, i'm used to it by now 🙂",
                    "perfect, thanks for the heads up. as always.", "wow shocking. it's fine.",
                    "cool cool cool. totally fine.", "sure, no problem. not like i was waiting.",
                    "amazing. thanks so much 🙂", "great, love waiting around for you haha"]
LATE_COLD_OVERT = ["unbelievable. every single time", "you're always late, it's so disrespectful",
                   "i'm leaving then. don't bother", "seriously?? i'm so sick of this", "that's so selfish of you"]

# --- busy-week decoy (warm, lower volume) ---------------------------------------------------------------
BUSY_WARM = ["crazy week at work, sorry i'm slow to answer ❤️", "swamped today but thinking of you",
             "sorry, drowning in {work} this week. miss you", "super busy, talk tonight? love you"]

# --- conflict episode (decoy days, and frequent in the overt-cold state) ----------------------------
CONFLICT_OPEN = [
    "why didn't you text me back all day?", "you said you'd call and you didn't",
    "i feel like you never listen to me", "you made plans without asking me again",
    "that comment at dinner really hurt", "are we going to talk about last night?",
]
CONFLICT_DEFEND = [
    "i was busy, i can't be on my phone 24/7", "that's not fair, i did try", "you're overreacting",
    "i don't know what you want from me", "i said i was sorry already", "can we not do this right now",
]
CONFLICT_ESCALATE = [
    "you always do this", "this is exactly the problem", "wow ok, forget it then",
    "i'm so tired of having the same fight",
]
REPAIR = [
    "i'm sorry, i didn't mean to snap at you. can we talk tonight?", "you're right, i should have texted. i'm sorry ❤️",
    "i hate fighting with you. i love you", "let's reset. dinner on me tonight?",
    "sorry for earlier. long day, not your fault", "i understand why you're upset. i'll do better",
]
REPAIR_ACCEPT = ["thank you. i love you too", "ok ❤️ i'm sorry too", "yes please. let's talk", "deal. i'm sorry too"]
