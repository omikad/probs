#include "infra/uci_impl.h"

using namespace std;


namespace probs {


UciImpl::UciImpl(ConfigParser& config) : uci_player(config) {
    cout << "id name Probs v1.0" << "\n";
    cout << "id author Sergey P" << "\n";
    cout << "uciok" << "\n";
    cout << flush;
}


void UciImpl::Run() {
    torch::NoGradGuard no_grad;

    while (true) {
        string input_line;
        getline(cin, input_line);

        stringstream line_stream(input_line);
        string command;
        line_stream >> command;

        if (command == "isready") {
            uci_player.WaitForReadyState();
            cout << "readyok" << "\n";
            cout << flush;
        }
        else if (command == "ucinewgame") {
            uci_player.OnNewGame();
        }
        else if (command == "position") {
            OnPositionCommand(line_stream);
        }
        else if (command == "go") {
            OnGoCommand(line_stream);
        }
        else if (command == "stop") {
            uci_player.Stop();
        }
        else if (command == "quit") {
            uci_player.Stop();
            break;
        }
        else if (command == "debug") {
            string on_off;
            line_stream >> on_off;
            if (on_off == "on")
                uci_player.SetDebug(true);
            else
                uci_player.SetDebug(false);
        }
        else if (command == "register") {
            cerr << "TODO: " << command << endl;
        }
        else if (command == "setoption") {
            cerr << "TODO: " << command << endl;
        }
        else if (command == "ponderhit") {
            cerr << "TODO: " << command << endl;
        }
        else if (command.empty()) {
            continue;
        }
        else {
            cerr << "Warning: Ignoring unknown command: '" << command << "'" << endl;
        }
    }
}


void UciImpl::OnPositionCommand(stringstream& line_stream) {
    string pos_string;
    vector<string> moves;
    bool is_reading_moves = false;
    while (line_stream) {
        string token;
        line_stream >> token;
        if (token.size() == 0)
            break;
        if (is_reading_moves)
            moves.push_back(token);
        else if (token == "startpos")
            pos_string = lczero::ChessBoard::kStartposFen;
        else if (token == "moves")
            is_reading_moves = true;
        else
            cerr << "WARNING: 'position' command unexpected token: '" << token << "'" << endl;
    }
    uci_player.SetPosition(pos_string, moves);
}


void UciImpl::OnGoCommand(stringstream& line_stream) {
    SearchConstraintsInfo search_info;
    search_info.wtime = nullopt;
    search_info.btime = nullopt;
    search_info.winc = nullopt;
    search_info.binc = nullopt;
    search_info.moves_to_go = nullopt;
    search_info.depth = nullopt;
    search_info.nodes = nullopt;
    search_info.mate = nullopt;
    search_info.fixed_time = nullopt;
    search_info.infinite = false;
    search_info.moves.clear();

    bool is_reading_moves = false;

    while (line_stream) {
        string token;
        line_stream >> token;
        if (token.size() == 0)
            break;
        if (token == "searchmoves")
            is_reading_moves = true;
        else if (token == "ponder") {
            cerr << "TODO: go ponder not supported" << endl;
        }
        else if (token == "wtime") {
            int x;
            line_stream >> x;
            search_info.wtime = std::chrono::milliseconds(x);
            is_reading_moves = false;
        }
        else if (token == "btime") {
            int x;
            line_stream >> x;
            search_info.btime = std::chrono::milliseconds(x);
            is_reading_moves = false;
        }
        else if (token == "winc") {
            int x;
            line_stream >> x;
            search_info.winc = std::chrono::milliseconds(x);
            is_reading_moves = false;
        }
        else if (token == "binc") {
            int x;
            line_stream >> x;
            search_info.binc = std::chrono::milliseconds(x);
            is_reading_moves = false;
        }
        else if (token == "movestogo") {
            int x;
            line_stream >> x;
            search_info.moves_to_go = x;
            is_reading_moves = false;
        }
        else if (token == "depth") {
            int x;
            line_stream >> x;
            search_info.depth = x;
            is_reading_moves = false;
        }
        else if (token == "nodes") {
            int x;
            line_stream >> x;
            search_info.nodes = x;
            is_reading_moves = false;
        }
        else if (token == "mate") {
            int x;
            line_stream >> x;
            search_info.mate = x;
            is_reading_moves = false;
        }
        else if (token == "movetime") {
            int x;
            line_stream >> x;
            search_info.fixed_time = std::chrono::milliseconds(x);
            is_reading_moves = false;
        }
        else if (token == "infinite") {
            search_info.infinite = true;
            is_reading_moves = false;
        }
        else if (is_reading_moves) {
            search_info.moves.push_back(token);
        }
        else
            cerr << "WARNING: 'go' command unexpected token: '" << token << "'" << endl;
    }

    uci_player.StartSearch(search_info);
}


}  // namespace probs
