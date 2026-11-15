"""What a healthy model does: the sequences of the health check and their floors.

A forward can equal the reference tensor for tensor and still not be the
model, when both read the same wrong bytes. These are checks of the model
against the world instead:

- `hello`: a request whose reply at temperature 0 is publicly reported. The
  upstream model repository's discussion #33 ("Non-thinking mode corrupts
  replies to short greetings when a system prompt is present") reports this
  reply, byte for byte the same, from the vendor's hosted API and from a
  self-hosted SGLang deployment of this checkpoint, and reports the three most
  likely tokens after "How can" as a newline, `#` and `<` at about -2.4 each.
  That is a third party's report, not a measurement made here; the reference
  on the hash-verified checkpoint reproduces every token of it, with -2.23,
  -2.34 and -2.69 at that position. A tie that close can fall either way
  under another arithmetic, so the check holds that one position to the three
  reported tokens and every other position to the reply's own.
- `story`: original prose as the assistant's turn of a chat request, scored
  by next-token accuracy. Public GPT-2 small and large score 0.342 and 0.397
  on the same text unprompted; the reference scores 0.665.
"""

SYSTEM = "You are a helpful assistant."

HELLO_MESSAGES = [("system", SYSTEM), ("user", "hello")]
# The reported reply, repeated word and all.
HELLO_REPLY = "Hello! How can\n can I help you today?"
# The reply up to the position whose three most likely tokens are reported.
HELLO_FORK = "Hello! How can"
HELLO_FORK_TOP3 = ("\n", "#", "<")

STORY_MESSAGES = [("system", SYSTEM), ("user", "Write a short story about a doctor on a small island who is offered a job on the mainland.")]
# Next-token accuracy over the story as the assistant's turn. Measured: 0.665.
STORY_FLOOR = 0.55
# Next-token accuracy over the passage of `real_golden.TEXT` after the
# beginning-of-sequence token. Measured: 0.670 (GPT-2 small 0.380, large 0.431).
PASSAGE_FLOOR = 0.55

STORY = (
    "The ferry to the island ran twice a day in winter, and the second crossing was the one the islanders "
    "trusted least. By four in the afternoon the light had already begun to drain out of the sky, and the "
    "wind that came down the channel carried a cold that no coat seemed able to keep out. Marta had made the "
    "crossing more times than she could count, first as a child holding her father's hand, later as a student "
    "coming home for the holidays, and now as the only doctor the island had. She stood at the rail with her "
    "bag between her feet and watched the harbour wall slide past.\n\n"
    "There were eleven passengers that afternoon, which was a crowd for January. Two of them were strangers, "
    "a man and a woman in new walking boots who kept consulting a folded map as though the island might have "
    "moved since it was printed. The rest she knew by name. Old Tomas sat inside by the heater with a crate of "
    "seed potatoes on his knees. The schoolteacher was marking exercise books with a red pencil, pausing now "
    "and then to shake her head. A boy of about twelve was trying to teach a dog to sit, and the dog, which "
    "was older and wiser than the boy, was pretending not to understand.\n\n"
    "The engine changed its note as they cleared the breakwater. Marta felt the deck lift and settle under "
    "her, the long slow movement of open water, and she tightened her grip on the rail without thinking about "
    "it. Her father used to say that the sea had only two moods, patient and impatient, and that a sensible "
    "person learned to tell them apart before leaving the harbour. Today it was patient. The swell came in "
    "low grey ridges, evenly spaced, and the ferry rode over them like a cart on a rutted road.\n\n"
    "She thought about the letter in her pocket. It had arrived that morning from the hospital on the "
    "mainland, and it offered her a position she would once have given a great deal to be offered. The salary "
    "was nearly twice what the island council paid her. There would be colleagues, proper equipment, a "
    "laboratory down the corridor instead of a two-day wait for results to come back on the boat. She had "
    "read it three times at the kitchen table and then put it in her coat, and she had not yet decided what "
    "she would say.\n\n"
    "The trouble was that she knew exactly what would happen if she left. The council would advertise the "
    "post, and nobody would apply, because nobody ever did. A nurse would come over on Tuesdays and "
    "Fridays. Anyone who fell ill on a Wednesday would have to wait, or be carried down to the harbour and "
    "taken across, and in winter there were weeks when the ferry did not run at all. She had spent six years "
    "persuading people that it was worth coming to see her early, before a cough became pneumonia or a sore "
    "foot became something worse, and she did not like to think how quickly that habit would be lost.\n\n"
    "Halfway across, the schoolteacher came out on deck and stood beside her. For a while neither of them "
    "spoke. Then the schoolteacher said that she had heard about the letter, because on an island everybody "
    "heard about everything, and that she wanted Marta to know that nobody would blame her if she went. "
    "People would be sorry, but they would understand. Marta thanked her and said that she had not made up "
    "her mind. The schoolteacher nodded, as if that was the answer she had expected, and went back inside to "
    "her exercise books.\n\n"
    "The island came up out of the dusk slowly, first as a darker line on the horizon, then as a shape with "
    "edges, and at last as a place with lights in it. Marta counted them as she always did. There was the "
    "lamp at the end of the pier, the three windows of the harbour office, the long row of cottages along the "
    "shore road, and higher up, alone on the hill, the light in the surgery that she had left burning so "
    "that she would be able to find the door. She picked up her bag. Whatever she decided, she thought, she "
    "would not decide it tonight.\n\n"
)
