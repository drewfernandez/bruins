// Define a class to represent a structured context item
enum AudioSceneCategory {
  ANIMALS
  NATURAL_SOUND_SCENES 
  HUMAN_NON_SPEECH_SOUNDS 
  INTERIOR_DOMESTIC_SOUNDS 
  EXTERIOR_URBAN_SOUNDS
}

class LabeledAudioScene {
  rational string
  confidence float
  category AudioSceneCategory
}

function LabelAudioScene(window_labels: string[]) -> LabeledAudioScene {
  client LMstudio4BGemma

  prompt #"You are an expert categorizer of audio scenes. Given a list of descriptions of the audio samples contained in the scene, classify the audio scene into one several categories.


Think through this step by step, and provide a final answer in the format below.  Consider all of the possible categories and select the one that best fits the descriptions provided. If the descriptions do not fit any of the categories, return "UNKNOWN".  when talking about hte categories make sure to use the UPPERCASE enum of the category name.   Walk through considering all of the categories, and then provide your final answer in the format below.


----
{{ ctx.output_format }}

{{_.role("user")}}
Descriptions:
{% for label in window_labels %}
  {{ "- " + label }}
{% endfor %}

"#
}


// ANIMALS
test label_easy_animals_audio_scene { 
  functions [LabelAudioScene]
  args {
    window_labels [
      "A dog barking",
      "Birds chirping",
      "A cat meowing"
    ]
  }
  @@assert({{ this.category == "ANIMALS" }})
}

test label_verbose_animals_audio_scene {
  functions [LabelAudioScene]
  args {
    window_labels [
      "A playful puppy barking excitedly at passersby",
      "Melodic birds chirping in the early morning light",
      "A soft cat meowing repeatedly for attention"
    ]
  }
  @@assert({{ this.category == "ANIMALS" }})
}

test label_noisy_verbose_animals_audio_scene {
  functions [LabelAudioScene]
  args {
    window_labels [
      "A dog barking loudly with panting and distant howls from another dog",
      "Multiple birds chirping warbling melodies amid rustling leaves",
      "A cat meowing urgently with faint motor hum and occasional creaking floorboards"
    ]
  }
  @@assert({{ this.category == "ANIMALS" }})
}

// NATURAL_SOUND_SCENES
test label_easy_natural_sound_scenes_audio_scene {
  functions [LabelAudioScene]
  args {
    window_labels [
      "Rain falling",
      "Wind blowing",
      "Ocean waves"
    ]
  }
  @@assert({{ this.category == "NATURAL_SOUND_SCENES" }})
}

test label_verbose_natural_sound_scenes_audio_scene {
  functions [LabelAudioScene]
  args {
    window_labels [
      "A steady rainfall pattering on the forest canopy",
      "Soft gusts of wind rustling through tall grass",
      "Gentle ocean waves lapping against the shoreline"
    ]
  }
  @@assert({{ this.category == "NATURAL_SOUND_SCENES" }})
}

test label_noisy_verbose_natural_sound_scenes_audio_scene {
  functions [LabelAudioScene]
  args {
    window_labels [
      "Heavy rain pelting leaves with intermittent thunderclaps and howling wind",
      "Rustling wind gusts scattering branches with distant bird calls",
      "Roaring surf crashing against rocky cliffs under squawking seagulls"
    ]
  }
  @@assert({{ this.category == "NATURAL_SOUND_SCENES" }})
}

// HUMAN_NON_SPEECH_SOUNDS
test label_easy_human_non_speech_sounds_audio_scene {
  functions [LabelAudioScene]
  args {
    window_labels [
      "Applause",
      "Laughter",
      "Coughing"
    ]
  }
  @@assert({{ this.category == "HUMAN_NON_SPEECH_SOUNDS" }})
}

test label_verbose_human_non_speech_sounds_audio_scene {
  functions [LabelAudioScene]
  args {
    window_labels [
      "A crowd applauding enthusiastically after a performance",
      "A group of friends laughing heartily",
      "A single person coughing repeatedly"
    ]
  }
  @@assert({{ this.category == "HUMAN_NON_SPEECH_SOUNDS" }})
}
