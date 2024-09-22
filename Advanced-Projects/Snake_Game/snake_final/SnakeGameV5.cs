using System;
using System.Threading;

namespace Snake_Game_V1
{
    class Snake_V1
    {
        //%%%%%%%%%%%%%%%%%%     Paramètres publics %%%%%%%%%%%%%%%%%%%%%%%%%%%
        Random random = new Random();

        int Height = 30;
        int Width = 100;

        //Liste contenant les positions ( X et Y) de composantes de la queue 
        int[] X;
        int[] Y;

        int fruit_x;
        int fruit_y;

        int star_x;
        int star_y;

        int[] obstacle_x;
        int[] obstacle_y;
        int nb_obstacle;

        int compteur;
        int compteur2;

        int score;
        int taille;


        bool Game_Over;
        bool start_game;
        bool pause_ok;
        bool once_ok;

        string dir;
        string last_dir;

        ConsoleKeyInfo keyInfo;
        ConsoleKey consoleKey;

        ///// Partie 1: Lancement Jeu et partie

        static void Main()
        {
            Snake_V1 snake = new Snake_V1();
            snake.Run();
        }
        void Game_partie()
        {
            Check_key_input();
            compteur += 1;

            Eat_fruit();
            Eat_star();
            Deplacement_Snake();
            Draw_fruit();
            Crash(X, Y, taille);
            Draw_snake();
            Affichage_score();
            Affiche_regle_jeu();
            Apparition_disparition_star();
            Apparitions_obstacles_multiples();
            Thread.Sleep(58);

        }
        public void Run()
        {
            Parametres_init();
            First_Window();
            Console.Clear();
            Draw_board();
            if (start_game)
            {
                while (!Game_Over)
                {

                    Game_partie();
                }
            }
            if (Game_Over)
            {
                End_Game();

            }
        }

        ///// Partie 2: Menus et initialisations
        public void Parametres_init()
        {
            score = 3;
            taille = 3;

            dir = "START";
            last_dir = "";

            X = new int[100];
            Y = new int[100];

            X[0] = Width / 2;
            Y[0] = Height / 2;
            X[1] = Width / 2;
            Y[1] = (Height / 2) + 1;
            X[2] = Width / 2;
            Y[2] = (Height / 2) + 2;

            fruit_x = random.Next(1, Width - 1);
            fruit_y = random.Next(1, Height - 1);


            Game_Over = false;
            start_game = false;
            pause_ok = false;
            once_ok = false;

            star_x = random.Next(1, Width - 1);
            star_y = random.Next(1, Height - 1);

            obstacle_x = new int[100];
            obstacle_y = new int[100];
            //obstacle_x[0] = random.Next(1, Width - 1);
            //obstacle_y[0] = random.Next(1, Height - 1);
            nb_obstacle = 1;

            compteur = 0;
            compteur2 = 0;

            Console.CursorVisible = false;
        }

        void First_Window()
        {


            Console.SetWindowPosition(0, 0);
            Console.SetWindowSize(Width + 20, Height + 10);
            Console.Title = "Snake Game";
            Console.ForegroundColor = ConsoleColor.Red;
            Console.CursorVisible = false;
            Console.SetCursorPosition(0, 5);
            Console.WriteLine("*********************************************************************************************************");
            Console.WriteLine("*                                                                                                       *");
            Console.WriteLine("*                                            SNAKE GAME                                                 *");
            Console.WriteLine("*                                                                                                       *");
            Console.WriteLine("*********************************************************************************************************" + "\n" + "\n");
            Console.WriteLine("                            APPUYER SUR UNE TOUCHE POUR LANCER LE JEU" + "\n" + "\n");
            Console.ForegroundColor = ConsoleColor.Yellow;
            Console.WriteLine("                            Credits: BELARBI, DE VILLELONGUE, MABILEAU" + "\n" + "\n");
            Console.ForegroundColor = ConsoleColor.White;
            Console.WriteLine("Commandes : Utilisez les fleches pour déplacer le serpent");
            Console.WriteLine("            Appuyer sur la toucher Echap pour terminer la partie ou quitter la fenetre du Menu");
            Console.WriteLine("            Appuyer sur la touche Espace pour mettre en pause la partie");
            Console.WriteLine("            La tête du serpent ne doit pas toucher ni la bordure blanche, ni une partie de son corps ni un carré bleu");
            Console.WriteLine("            Si la tête du serpent touche un fruit F, le serpent s'allonge d'un élément, le score augmente d'un point");
            Console.WriteLine("            Si la tête du serpent touche une étoile * le serpent s'allonge de 3 éléments, le score augmente de 3 points");
            Console.WriteLine("            Le score s'affiche en dessous de la zone de jeu, il correspond à la taille de votre serpent");
            Console.WriteLine("            La taille de votre serpent est le nombre d'éléments qui le compose, tête comprise");


            keyInfo = Console.ReadKey(true);
            consoleKey = keyInfo.Key;

            if (consoleKey == ConsoleKey.Escape)
            { Environment.Exit(0); }
            else
            {
                start_game = true;
            }
        }

        void End_Game()
        {
            Console.Clear();
            Console.ForegroundColor = ConsoleColor.Red;
            Console.SetCursorPosition(0, 5);
            Console.WriteLine("                                  ***   GAME OVER   ***" + "\n");
            Console.WriteLine($"                                  Votre score est de : {taille} " + "\n");
            Console.WriteLine("         Vous avez perdu, veuillez appuyer sur Echap pour quitter ou Entrée pour rejouer");


            while (true)
            {
                keyInfo = Console.ReadKey(true);

                if (keyInfo.Key == ConsoleKey.Escape)
                { Environment.Exit(0); }
                if (keyInfo.Key == ConsoleKey.Enter)
                {
                    Run();
                    break;
                }
            }

        }

        void Pause()
        {
            while (pause_ok)
            {
                Console.Clear();
                Console.SetCursorPosition(Width / 2 - 6, Height / 2);
                Console.WriteLine("GAME PAUSED" + "\n" + "\n");
                Console.WriteLine("Appuyer sur Espace ou Entrée pour reprendre la partie");
                Console.WriteLine("Appuyer sur Backspace/Retour en arriere pour pour lancer une nouvelle partie");
                Console.WriteLine("Appuyer sur Echap pour quitter la partie");

                keyInfo = Console.ReadKey(true);
                consoleKey = keyInfo.Key;
                if (consoleKey == ConsoleKey.Escape)
                {
                    Game_Over = true;
                    End_Game();
                    break;
                }
                if (consoleKey == ConsoleKey.Spacebar || consoleKey == ConsoleKey.Enter)
                {

                    pause_ok = false;
                    Console.Clear();
                    Draw_board();
                    break;
                }
                if (consoleKey == ConsoleKey.Backspace)
                {
                    Run();
                    break;
                }

            }
            dir = last_dir;
        }

        ///// Partie 3: Affichage

        void Affichage_score()
        {
            Console.ForegroundColor = ConsoleColor.Red;
            Console.SetCursorPosition(Width / 2 - 3, Height + 2);
            Console.WriteLine($"Score: {score} ");
        }

        void Affiche_regle_jeu()
        {
            Console.ForegroundColor = ConsoleColor.White;
            Console.SetCursorPosition(0, Height + 4);
            Console.WriteLine("Instructions: ");
            Console.WriteLine("             Appuyer sur une flèche (sauf bas) pour mettre en mouvement le serpent");
            Console.WriteLine("             Vous ne pouvez pas débuter avec la flèche du bas car la tête du serpent est orientée vers le haut");
            Console.WriteLine("             Appuyer sur la touche Echap pour terminer la partie");
            Console.WriteLine("             Appuyer sur la touche Espace pour mettre en pause la partie");
        }

        void Draw_snake()
        {
            ConsoleColor c = ConsoleColor.Green;
            //Tête du serpent
            Draw_char('O', X[0], Y[0], c);
            //Corps du serpent
            for (int i = 1; i <= (taille - 1); i++)
            {
                Draw_char('o', X[i], Y[i], c);
            }
            //Permet d'effacer le derner charactère: garder la taille constante + mvmt
            if (X[taille] != 0)
            {
                Console.SetCursorPosition(X[taille], Y[taille]);
                Console.Write("\0");
            }
        }

        void Draw_fruit()
        {
            ConsoleColor c = ConsoleColor.Red;
            Draw_char('F', fruit_x, fruit_y, c);
        }

        void Draw_star()
        {
            ConsoleColor c = ConsoleColor.Yellow;
            Draw_char('*', star_x, star_y, c);
        }

        void Draw_obstacle()
        {
            ConsoleColor c = ConsoleColor.DarkBlue;
            Draw_char('■', obstacle_x[nb_obstacle - 1], obstacle_y[nb_obstacle - 1], c);
        }

        void Draw_board()
        {

            ConsoleColor c = ConsoleColor.White;
            for (int i = 1; i <= (Width); i++) //Haut
            {
                Draw_char('-', i, 0, c);
            }
            for (int i = 1; i <= Width; i++) //Bas 
            {
                Draw_char('-', i, Height + 1, c);
            }
            for (int i = 1; i <= Height; i++) //Gauche
            {
                Draw_char('|', 0, i, c);
            }
            for (int i = 1; i <= Height; i++) //Droit
            {
                Draw_char('|', Width + 1, i, c);
            }

            //Coins
            Draw_char('┌', 0, 0, c);
            Draw_char('┐', Width + 1, 0, c);
            Draw_char('└', 0, Height + 1, c);
            Draw_char('┘', Width + 1, Height + 1, c);
        }

        public static void Draw_char(char car, int a, int b, ConsoleColor c)
        {
            Console.SetCursorPosition(a, b);
            Console.ForegroundColor = c;
            Console.Write(car);
        }

        ///// Partie 4: Déplacements du serpent

        void Check_key_input()
        {
            while (Console.KeyAvailable && dir != "START") // bool indiquant si l'appui sur une touche est disponible dans le flux d'entrée
            {

                keyInfo = Console.ReadKey(true);
                consoleKey = keyInfo.Key;

                if (consoleKey == ConsoleKey.Escape)
                {
                    Game_Over = true;
                    break;
                }

                if (consoleKey == ConsoleKey.Spacebar)
                {
                    last_dir = dir;
                    dir = "PAUSE";
                    pause_ok = true;

                }
                else if (consoleKey == ConsoleKey.LeftArrow)
                {
                    last_dir = dir;
                    dir = "LEFT";
                }
                else if (consoleKey == ConsoleKey.RightArrow)
                {
                    last_dir = dir;
                    dir = "RIGHT";
                }
                else if (consoleKey == ConsoleKey.UpArrow)
                {
                    last_dir = dir;
                    dir = "UP";
                }
                else if (consoleKey == ConsoleKey.DownArrow)
                {
                    last_dir = dir;
                    dir = "DOWN";
                }
            }
        }

        void First_mvmt()
        {
            while (dir == "START")
            {
                Draw_fruit();
                Crash(X, Y, taille);
                Draw_snake();
                Affichage_score();
                Affiche_regle_jeu();

                while (Console.KeyAvailable)
                {
                    keyInfo = Console.ReadKey(true);
                    consoleKey = keyInfo.Key;
                    if (consoleKey == ConsoleKey.LeftArrow)
                    { dir = "LEFT"; }
                    if (consoleKey == ConsoleKey.RightArrow)
                    { dir = "RIGHT"; }
                    if (consoleKey == ConsoleKey.UpArrow)
                    { dir = "UP"; }
                }
                break;
            }
        }

        void Mouvement_corps_snake()
        {
            for (int i = taille + 1; i > 1; i--)
            {
                X[i - 1] = X[i - 2];
                Y[i - 1] = Y[i - 2];
            }
        }

        void Deplacement_Snake()
        {

            switch (dir)
            {
                case "START":
                    First_mvmt();
                    break;

                case "UP":
                    if (last_dir == "DOWN")
                    {
                        dir = "DOWN";
                    }
                    else
                    {
                        Mouvement_corps_snake();
                        Y[0] -= 1;
                    }
                    break;

                case "DOWN":
                    if (last_dir == "UP")
                    {
                        dir = "UP";

                    }
                    else
                    {
                        Mouvement_corps_snake();
                        Y[0] += 1;
                    }
                    break;

                case "LEFT":
                    if (last_dir == "RIGHT")
                    {
                        dir = "RIGHT";
                    }
                    else
                    {
                        Mouvement_corps_snake();
                        X[0] -= 1;
                    }
                    break;

                case "RIGHT":
                    if (last_dir == "LEFT")
                    {
                        dir = "LEFT";
                    }
                    else
                    {
                        Mouvement_corps_snake();
                        X[0] += 1;
                    }
                    break;

                case "PAUSE":
                    Pause();
                    break;
            }
        }

        ///// Partie 5: Eléments (fruits, etoile, obstacle)

        void New_fruit_position()
        {
            fruit_x = random.Next(1, Width);
            fruit_y = random.Next(1, Height);

            //Gestion du cas où le fruit se trouve sur le corps su serpent

            for (int i = taille; i >= 1; i--)
            {
                if (X[i - 1] == fruit_x && Y[i - 1] == fruit_y)
                {
                    New_fruit_position();
                }
            }
        }

        void New_star_position()
        {
            star_x = random.Next(1, Width);
            star_y = random.Next(1, Height);

            //Gestion du cas où l'étoile se trouve sur le corps su serpent

            for (int i = taille; i >= 1; i--)
            {
                if (X[i - 1] == star_x && Y[i - 1] == star_y)
                {
                    New_star_position();
                }
            }
        }

        void Apparition_disparition_star()
        {
            if (compteur >= 300)
            {
                Draw_star();
            }
            if (compteur >= 400)
            {
                Console.SetCursorPosition(star_x, star_y);
                Console.Write("\0");
                New_star_position();
                compteur = 0;
            }
            else if (once_ok)
            {
                Draw_char('┌', 0, 0, ConsoleColor.White);
                once_ok = false;
            }

        }

        void New_obstacle_position()
        {
            obstacle_x[nb_obstacle - 1] = random.Next(1, Width);
            obstacle_y[nb_obstacle - 1] = random.Next(1, Height);

            //Gestion du cas où l'obstacle se trouve sur le corps su serpent

            for (int i = taille; i >= 1; i--)
            {
                for (int j = 0; j < nb_obstacle; j++)
                {
                    if (X[i - 1] == obstacle_x[j] && Y[i - 1] == obstacle_y[j])
                    {
                        New_obstacle_position();
                    }
                }
            }
        }

        void Apparitions_obstacles_multiples()
        {
            if (score >= 10)
            {
                compteur2 += 1;
                if (compteur2 >= 200)
                {
                    New_obstacle_position();
                    Draw_obstacle();
                    nb_obstacle += 1;
                    compteur2 = 0;
                }

            }
        }

        ///// Partie 6: Interactions avec éléments (Fruit, Etoile, Obstacle, Barrière, Lui même)

        void Eat_fruit()
        {
            if (X[0] == fruit_x && Y[0] == fruit_y)
            {
                taille += 1;
                score += 1;
                New_fruit_position();
            }
        }

        void Eat_star()
        {
            if (X[0] == star_x && Y[0] == star_y)
            {
                taille += 3;
                score += 3;
                compteur = 0;
                New_star_position();
                once_ok = true;
            }
        }

        void Crash(int[] X, int[] Y, int taille)
        {
            // sortie de la zone de jeu
            if (X[0] <= 0 || X[0] >= Width + 1
                || Y[0] <= 0 || Y[0] >= Height + 1)
            {
                Game_Over = true;
            }
            // collision avec la queue
            for (int i = taille; i >= 2; i--)
            {
                if (X[0] == X[i - 1] && Y[0] == Y[i - 1])
                {
                    Game_Over = true;
                }
            }
            //collision avec un obstacle
            for (int j = 0; j < nb_obstacle; j++)
            {
                if (X[0] == obstacle_x[j] && Y[0] == obstacle_y[j])
                {
                    Game_Over = true;
                }
            }

        }
    }
}

