# Install developpement environnement

> python version : 3.9.19 ?

With arch, use packages `pyenv` and `pyenv-virtualenv`.

```sh
pyenv install 3.9.19
pyenv global 3.9.19
pyenv virtualenv .env_bs11
source ~/.pyenv/versions/3.9.19/envs/.env_bs11/bin/activate
pip install -r qrequirements.txt
```

# Start in developpement environnement

```sh
./start.sh
```

TODO: Add link to drive-dataset and test_set used !

# Untracked files

Missing folder `./labos/drive_dataset` should contain :

```
.
├── 0
│   ├── 0_Soul Blues Electric Guitar Instrumental.mp3
│   ├── 1_Blues.mp3
│   ├── Ben Harper - Need To Know Basis_UXv53Nq8g_E.mp3
│   └── Wait Til I Get Over_0xhaweCEEzE.mp3
├── 1
│   ├── 0_Nocturne op.9 No.2.mp3
│   ├── 1_Serenade.mp3
│   ├── Antonio Vivaldi - Concerto No.4 in F minor, Op.8, RV 297, _ L_inverno _, Allegro Non Molto_NzCL9uLkQSI.mp3
│   └── Concerto Grosso for Strings _Palladio__ I. Allegretto_xUi0mM6-PJ0.mp3
├── 2
│   ├── 0_Achy Breaky Heart.mp3
│   ├── 1_Kirin J Callinan Big Enough ft Alex Cameron Molly, Lewis Jimmy Barnes.mp3
│   ├── 1_When It Rains It Pours.mp3
│   └── Willie Nelson - Live Forever.mp3
├── 3
│   ├── 0_Lets Groove.mp3
│   ├── 1_Don’t Stop Til You Get Enough.mp3
│   ├── Daft Punk - Get Lucky (Official Audio) ft. Pharrell Williams, Nile Rodgers_5NV6Rdv1a3I.mp3
│   └── Kool _ The Gang - Get Down On It_qchPLaiKocI.mp3
├── 4
│   ├── 1_Argent Drogue Sexe.mp3
│   ├── 1_Dj Abdel - Funky Cops  Lets boogie.mp3
│   ├── ALKPOTE feat. VALD _  #EP1 - PLUS HAUT - Les Marches De l_Empereur Saison 3_T_X-BbsXXAk.mp3
│   └── Freeze Corleone 667 - Freeze Raël.mp3
├── 5
│   ├── Cuphead OST - Inkwell Isle Three [Music]_BVP65Rg8myE.mp3
│   ├── Frank_Sinatra_-_Fly_Me_To_The_Moon_Live_At_The_Kiel_Opera_House_St._Louis_MO_1965_Y2rDb4Ur2dw.mp3
│   ├── La vie en rose - Louis Armstrong_8IJzYAda1wA.mp3
│   └── Miles_Davis_Quintet_-_Round_Midnight_GIgLt7LAZF0.mp3
├── 6
│   ├── 0_Enter Sandman.mp3
│   ├── 1_Bob léponge - Glouton Barjot.mp3
│   ├── 1_Walk.mp3
│   └── Pentakill_ Mortal Reminder _ Official Music Video - League of Legends_5-mT9D4fdgQ.mp3
├── 7
│   ├── 0_Roar.mp3
│   ├── 1_Never Forget You.mp3
│   ├── Ed Sheeran - Shape of You (Official Music Video)_JGwWNGJdvx8.mp3
│   └── OneRepublic - RUNAWAY (Official Music Video)_qWJU_eANW4M.mp3
├── 8
│   ├── Damian__Jr._Gong__Marley_-_Welcome_To_Jamrock_Official_Video__GZlJGERbvE.mp3
│   ├── Lutan Fyah - Rasta Reggae Music (Official Music Video)_be67OhUSULg.mp3
│   ├── onlymp3.to - Kabaka Pyramid The Kalling ft. Stephen Marley Protoje Jesse Royal Official Music Video -nf0itXfkHHo-192k-1693975416.mp3
│   └── Protoje_-_Who_Knows_ft._Chronixx_Official_Music_Video_hzqFmXZ8tOE.mp3
└── 9
    ├── 0_Canon Rock Final.mp3
    ├── 1_Roxanne.mp3
    ├── AC_DC Back in Black.mp3
    └── Numb (Official Music Video) [4K UPGRADE] – Linkin Park_kXYiU_JCYtU.mp3
```

Missing folder `./labos/test_set` should contain :

```
.
├── 0
│   ├── Blues_ Brandon Lane - Trouble_TTfVNYJxXw8.mp3
│   ├── Chris Bell - Elevator To Heaven_5ODL5_djyBI.mp3
│   └── Detroit Blues Band – Walkin_ Out The Door_Hkt1yULpfRE.mp3
├── 1
│   ├── Bach - Toccata in D Minor_HQnALzuCGSQ.mp3
│   ├── Dmitri Shostakovich - Waltz No. 2_mmCnQDUSO4I.mp3
│   └── Tchaikovsky - Swan Lake (Act II, No. 10)_do6Ki6kMq_o.mp3
├── 2
│   ├── Johnny Cash, June Carter - Jackson (Official Audio)_GPLKCGTkzgo.mp3
│   ├── Maddie _ Tae - Friends Don_t (Official Music Video)_XGmJMvnDZEg.mp3
│   └── Tim McGraw - One Of Those Nights.mp3
├── 3
│   ├── Hamilton Bohannon - Me And The Gang (Extended Version).mp3
│   ├── Village People - YMCA OFFICIAL Music Video 1978_CS9OO0S5w2k.mp3
│   └── Viola Wills - gonna get along without you now (lp) original version (1979)__JALXA3NgsU.mp3
├── 4
│   ├── Annihilate (Spider-Man_ Across the Spider-Verse)_dsnuu20RSFU.mp3
│   ├── Kendrick Lamar - N95_zI383uEwA6Q.mp3
│   └── POP SMOKE - DIOR (OFFICIAL VIDEO)_oorVWW9ywG0.mp3
├── 5
│   ├── Ray Charles - Georgia On My Mind (Official Video)_ggGzE5KfCio.mp3
│   ├── _SING, SING, SING_ BY BENNY GOODMAN_r2S1I_ien6A.mp3
│   └── Tony Bennett - The Good Life (Original) HQ 1963.mp3
├── 6
│   ├── BLACK SABBATH - _Iron Man_ (Official Video)_5s7_WbiR79E.mp3
│   ├── Bury the Light - Vergil_s battle theme from Devil May Cry 5 Special Edition_Jrg9KxGNeJY.mp3
│   └── Disturbed - Down With The Sickness (Official Music Video) [HD UPGRADE]_09LTT0xwdfw.mp3
├── 7
│   ├── Angèle feat. Roméo Elvis - Tout Oublier [CLIP OFFICIEL]_Fy1xQSiLx8U.mp3
│   ├── Stromae - papaoutai (Official Video)_oiKj0Z_Xnjc.mp3
│   └── Vianney - Je m_en vais [Clip officiel]_eLYyCFuPCX8.mp3
├── 8
│   ├── Black Uhuru - Guess whos coming to dinner_KWEGXb2juvM.mp3
│   ├── Buju Banton - Blessed_tzmzKtOaVf4.mp3
│   └── Johnny Osbourne - Truths And Rights (Extended Mix)  1979_FAxdYIsXu-s.mp3
└── 9
    ├── Nirvana - Smells Like Teen Spirit (Official Music Video)_hTWKbfoikeg.mp3
    ├── Skillet - Monster (Official Video)_1mjlM_RnsVE.mp3
    └── System Of A Down - Chop Suey_ (Official HD Video)_CSvFpBOe8eY.mp3

```
